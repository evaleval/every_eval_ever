(() => {
  const searchIndexUrl = document.body.dataset.searchIndex;
  const inputs = Array.from(document.querySelectorAll('[data-search-input]'));
  const resultBoxes = Array.from(
    document.querySelectorAll('[data-search-results]')
  );
  let indexPromise;

  const escapeHtml = (value) =>
    String(value).replace(/[&<>"']/g, (char) => {
      const entities = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
      };
      return entities[char];
    });

  const normalize = (value) =>
    String(value).toLowerCase().replace(/\s+/g, ' ').trim();

  const clearResults = () => {
    resultBoxes.forEach((box) => {
      box.innerHTML = '';
    });
  };

  const loadIndex = () => {
    if (!indexPromise) {
      indexPromise = fetch(searchIndexUrl, { cache: 'force-cache' }).then(
        (response) => {
          if (!response.ok) {
            throw new Error(`Search index failed: ${response.status}`);
          }
          return response.json();
        }
      );
    }
    return indexPromise;
  };

  const scoreItem = (item, tokens) => {
    const title = normalize(item.title);
    const headings = (item.headings || []).map(normalize);
    const haystack = normalize(
      `${item.title} ${item.route} ${headings.join(' ')} ${item.summary || ''}`
    );

    return tokens.reduce((score, token) => {
      if (title.includes(token)) {
        return score + 4;
      }
      if (headings.some((heading) => heading.includes(token))) {
        return score + 2;
      }
      return haystack.includes(token) ? score + 1 : score;
    }, 0);
  };

  const renderResults = (items, query) => {
    const normalized = normalize(query);
    if (normalized.length < 2) {
      clearResults();
      return;
    }

    const tokens = normalized.split(' ');
    const matches = items
      .map((item) => ({ item, score: scoreItem(item, tokens) }))
      .filter((entry) => entry.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 6);

    const html = matches.length
      ? matches
          .map(
            ({ item }) =>
              `<a class="search-result" href="${escapeHtml(item.route)}">` +
              `<strong>${escapeHtml(item.title)}</strong>` +
              `<span>${escapeHtml(item.summary || '')}</span>` +
              '</a>'
          )
          .join('')
      : '<div class="search-result"><strong>No matches</strong><span>Try schema, converter, validation, or datastore.</span></div>';

    resultBoxes.forEach((box) => {
      box.innerHTML = html;
    });
  };

  inputs.forEach((input) => {
    input.addEventListener('input', () => {
      loadIndex()
        .then((items) => renderResults(items, input.value))
        .catch(() => {
          resultBoxes.forEach((box) => {
            box.innerHTML =
              '<div class="search-result"><strong>Search unavailable</strong><span>The local index could not be loaded.</span></div>';
          });
        });
    });
  });
})();
