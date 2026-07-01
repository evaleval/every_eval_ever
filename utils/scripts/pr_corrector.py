import os
import re
import json
import shutil
import stat
import urllib.request
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

from every_eval_ever.validate import validate_file

# --- CONFIGURATION ---
REPO_ID = "evaleval/EEE_datastore"
REPO_TYPE = "dataset"
WORKSPACE = "EEE_Targeted_Workspace"
SCHEMA_PATH = "eval.schema.json"


def remove_readonly(func, path, excinfo):
    """Clear the read-only bit on Windows so shutil can delete files."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def load_schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_ref(schema, ref):
    parts = ref.replace("#/", "").split("/")
    current = schema
    for part in parts:
        current = current.get(part, {})
    return current


def get_schema_info(loc_string, root_schema):
    """Parses loc string like 'evaluation_results -> [0] -> metric_config' and returns its schema."""
    parts = loc_string.split(" -> ")
    current = root_schema
    
    for part in parts:
        if "$ref" in current:
            current = resolve_ref(root_schema, current["$ref"])
            
        if part.startswith("[") and part.endswith("]"):
            current = current.get("items", {})
        else:
            if "properties" in current and part in current["properties"]:
                current = current["properties"][part]
            elif "oneOf" in current:
                found = False
                for option in current["oneOf"]:
                    opt = option
                    if "$ref" in opt:
                        opt = resolve_ref(root_schema, opt["$ref"])
                    if "properties" in opt and part in opt["properties"]:
                        current = opt["properties"][part]
                        found = True
                        break
                if not found:
                    current = {}
            else:
                current = {}
                
    if "$ref" in current:
        current = resolve_ref(root_schema, current["$ref"])
        
    return current


def convert_value(user_input, schema_info):
    if not schema_info:
        return user_input
        
    expected_type = schema_info.get("type")
    
    if expected_type == "boolean":
        return user_input.lower() in ["true", "1", "yes", "y", "t"]
    elif expected_type == "integer":
        try:
            return int(user_input)
        except ValueError:
            return user_input
    elif expected_type == "number":
        try:
            return float(user_input)
        except ValueError:
            return user_input
    elif expected_type == "array":
        try:
            parsed = json.loads(user_input)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        # Simple fallback for arrays
        return [s.strip() for s in user_input.split(",") if s.strip()]
    
    # If type is an array of options e.g., ["null", "number"]
    if isinstance(expected_type, list):
        if "number" in expected_type:
            try:
                return float(user_input)
            except ValueError:
                pass
        if "integer" in expected_type:
            try:
                return int(user_input)
            except ValueError:
                pass
                
    return user_input


def apply_fix_exact(data, loc_string, value):
    """Dynamically traverses a JSON object and injects the value at the exact location."""
    parts = loc_string.split(" -> ")
    if not parts:
        return

    current = data
    for i in range(len(parts) - 1):
        part = parts[i]
        if part.startswith("[") and part.endswith("]"):
            index = int(part[1:-1])
            current = current[index]
        else:
            if part not in current:
                current[part] = {}
            current = current[part]

    last_part = parts[-1]
    if last_part.startswith("[") and last_part.endswith("]"):
        index = int(last_part[1:-1])
        current[index] = value
    else:
        current[last_part] = value


def apply_fix_fuzzy(data, path_str, key, value):
    """Dynamically traverses a JSON object using dot notation and injects the key/value pair."""
    parts = path_str.split('.') if path_str else []

    def traverse(current_obj, path_parts):
        if not path_parts:
            if isinstance(current_obj, dict):
                current_obj[key] = value
            return

        part = path_parts[0]
        if isinstance(current_obj, dict):
            if part not in current_obj:
                current_obj[part] = {}
            if len(path_parts) == 1:
                if isinstance(current_obj[part], list):
                    for item in current_obj[part]:
                        if isinstance(item, dict):
                            item[key] = value
                else:
                    current_obj[part][key] = value
            else:
                traverse(current_obj[part], path_parts[1:])
        elif isinstance(current_obj, list):
            for item in current_obj:
                traverse(item, path_parts)

    traverse(data, parts)


def get_pr_files(repo_id, pr_num, token):
    """Fetches the list of modified .json files in the PR using diffUrl or HTML scraping as a fallback."""
    url = f"https://huggingface.co/api/datasets/{repo_id}/discussions/{pr_num}"
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
        
    try:
        with urllib.request.urlopen(req) as res:
            data = json.loads(res.read().decode('utf-8'))
            
        diff_url = data.get('diffUrl')
        if diff_url:
            if diff_url.startswith('/'):
                diff_url = "https://huggingface.co" + diff_url
            diff_req = urllib.request.Request(diff_url)
            if token:
                diff_req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(diff_req) as diff_res:
                diff_text = diff_res.read().decode('utf-8')
                files = re.findall(r'^\+\+\+ b/(data/.*\.json)$', diff_text, flags=re.MULTILINE)
                if files:
                    return list(set(files))
    except Exception as e:
        pass
        
    # Fallback to HTML scraping
    html_url = f"https://huggingface.co/datasets/{repo_id}/discussions/{pr_num}/files"
    html_req = urllib.request.Request(html_url)
    if token:
        html_req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(html_req) as res:
            html = res.read().decode('utf-8')
            files = re.findall(r'(data/[\w\-\./]+\.json)', html)
            return list(set(files))
    except Exception as e:
        print(f"❌ Failed to parse files from PR: {e}")
        
    return []


def custom_validate(data):
    """Custom heuristic validator for warnings not strictly in the schema."""
    bot_warnings = []
    
    # 1. Check for deployment_type and dependent fields
    model_info = data.get('model_info', {})
    additional = model_info.get('additional_details', {})
    
    deployment_type = additional.get('deployment_type')
    if not deployment_type:
        bot_warnings.append(('model_info.additional_details', 'deployment_type', 'Expected one of: api, local, unknown'))
    else:
        if deployment_type == 'api' and 'model_availability' not in additional:
            bot_warnings.append(('model_info.additional_details', 'model_availability', "Expected one of: closed_source, open_weights_deployment, other"))
        
    return bot_warnings


def main():
    print("🤖 EEE Datastore Targeted PR Fixer 🤖\n")

    hf_token = input("Enter your Hugging Face Access Token: ").strip()
    pr_input = input("Enter the PR number or URL (e.g., 136 or https://.../136): ").strip()

    pr_num = re.search(r'(\d+)$', pr_input).group(1)
    revision = f"refs/pr/{pr_num}"
    api = HfApi(token=hf_token)

    try:
        schema = load_schema()
    except Exception as e:
        print(f"❌ Failed to load {SCHEMA_PATH}: {e}")
        return

    print("\n📂 Fetching file list for the PR...")
    json_files = get_pr_files(REPO_ID, pr_num, hf_token)

    if not json_files:
        print("❌ No aggregate JSON files found in this PR to fix. Is the PR number correct?")
        return
        
    print(f"Found {len(json_files)} file(s) in PR #{pr_num}.")

    if os.path.exists(WORKSPACE):
        shutil.rmtree(WORKSPACE, onexc=remove_readonly)
    os.makedirs(WORKSPACE)

    operations = []
    apply_to_all_answers = {}
    print(f"Downloading {len(json_files)} JSON file(s) locally...\n")

    for file_path in json_files:
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=file_path,
                revision=revision,
                token=hf_token,
                local_dir=WORKSPACE,
                local_dir_use_symlinks=False
            )
        except Exception as e:
            print(f"❌ Failed to download {file_path}: {e}")
            continue

        modified = False

        while True:
            # Validate the downloaded file
            report = validate_file(Path(local_path))
            
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            bot_warnings = custom_validate(data)
            missing_errors = [err for err in report.errors if "missing" in err.get("type", "")]
            
            if report.valid and not bot_warnings and not missing_errors:
                if not modified:
                    print(f"✅ {file_path.split('/')[-1]} passed validation and has no bot warnings! No fixes needed.")
                break

            if not missing_errors and not bot_warnings:
                print(f"⚠️ {file_path.split('/')[-1]} has non-missing validation errors:")
                for err in report.errors:
                    print(f"   - {err['loc']}: {err['msg']}")
                break

            print(f"\n📄 Fixing {file_path.split('/')[-1]} ({len(missing_errors)} missing fields, {len(bot_warnings)} bot warnings)")
            made_changes_this_round = False

            # Process standard validation missing errors
            for err in missing_errors:
                loc_str = err['loc']
                last_key = loc_str.split(" -> ")[-1]
                
                # Check for "apply all" override based on the key name
                if last_key in apply_to_all_answers:
                    apply_fix_exact(data, loc_str, apply_to_all_answers[last_key])
                    made_changes_this_round = True
                    continue

                schema_info = get_schema_info(loc_str, schema)
                desc = schema_info.get("description", "No description available")
                expected_type = schema_info.get("type", "unknown")
                
                print(f"\n   Missing: '{last_key}' at '{loc_str}'")
                print(f"   Type: {expected_type} | Description: {desc}")
                
                user_input = input(f"   Enter value (or 'skip', or 'all:your_value'): ").strip()

                if user_input.lower() == 'skip' or user_input == '':
                    print("   ⏭️ Skipped.")
                    continue

                if user_input.lower().startswith("all:"):
                    raw_value = user_input[4:].strip()
                    converted = convert_value(raw_value, schema_info)
                    apply_to_all_answers[last_key] = converted
                    apply_fix_exact(data, loc_str, converted)
                    made_changes_this_round = True
                    print(f"   ✅ Applied '{converted}' to this and all future occurrences of '{last_key}'.")
                else:
                    converted = convert_value(user_input, schema_info)
                    apply_fix_exact(data, loc_str, converted)
                    made_changes_this_round = True
                    print(f"   ✅ Applied '{converted}'.")

            # Process bot warnings
            for path_str, key, desc in bot_warnings:
                if key in apply_to_all_answers:
                    apply_fix_fuzzy(data, path_str, key, apply_to_all_answers[key])
                    made_changes_this_round = True
                    continue

                print(f"\n   Bot Warning Missing: '{key}' inside '{path_str}'")
                print(f"   Description: {desc}")
                user_input = input(f"   Enter value (or 'skip', or 'all:your_value'): ").strip()

                if user_input.lower() == 'skip' or user_input == '':
                    print("   ⏭️ Skipped.")
                    continue

                # Fallback for bot warning schema lookup is harder without exact array indices
                if user_input.lower().startswith("all:"):
                    raw_value = user_input[4:].strip()
                    # for bot warnings we just inject string directly or attempt basic inference
                    converted = convert_value(raw_value, {})
                    apply_to_all_answers[key] = converted
                    apply_fix_fuzzy(data, path_str, key, converted)
                    made_changes_this_round = True
                    print(f"   ✅ Applied '{converted}' to this and all future files.")
                else:
                    converted = convert_value(user_input, {})
                    apply_fix_fuzzy(data, path_str, key, converted)
                    made_changes_this_round = True
                    print(f"   ✅ Applied '{converted}'.")

            if made_changes_this_round:
                modified = True
                with open(local_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                    f.write("\n")
            else:
                # User skipped everything, break to avoid infinite loop
                break

        if modified:
            operations.append(CommitOperationAdd(path_in_repo=file_path, path_or_fileobj=local_path))

    if operations:
        print(f"\n📤 Committing {len(operations)} updated file(s) to PR #{pr_num}...")
        api.create_commit(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            revision=revision,
            operations=operations,
            commit_message="Fix schema warnings via interactive script"
        )
        print("🎉 Success! Your PR has been updated.")
    else:
        print("\n⚠️ No changes were made that require a commit.")

    print("🧹 Cleaning up local files...")
    shutil.rmtree(WORKSPACE, onexc=remove_readonly)
    print("✨ Cleanup complete!")


if __name__ == "__main__":
    main()