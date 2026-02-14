from typing import List
from helm.benchmark.adaptation.scenario_state import RequestState

def extract_reasoning(request_state: RequestState) -> str | None:
    if request_state.result and request_state.result.completions:
        return getattr(
            getattr(
                request_state.result.completions[0], 
                "thinking", 
                None
            ),
            "text",
            None
        )
    
    return None

def extract_all_reasonings(request_state: RequestState) -> List[str] | None:
    if not (request_state.result and request_state.result.completions):
        return None

    return [
        getattr(getattr(c, "thinking", None), "text", None)
        for c in request_state.result.completions
        if getattr(c, "thinking", None) is not None
    ]