// Search, loaded when a reader asks for it.
//
// mdBook's template loads elasticlunr, mark.js and searcher.js on every page,
// and searcher.js then fetches the whole search index (2.3 MB, ~190 KB with
// brotli) whether or not anyone searches. On a phone that download competes
// with the page's own first paint. index.hbs no longer links the three
// scripts; this file loads them the first time the reader opens search (the
// search button, or the `s` key), or straight away when the address already
// carries a search or a highlight (a search result opened in a new tab, a
// reload, back and forward).
//
// Until searcher.js has its index, the search bar opens and takes typing as
// before. searcher.js finishes starting by setting the search button's
// aria-expanded (it closes the bar, or opens it for a ?search= address); the
// first such write this file did not make is the signal to hand over: the bar
// is reopened through searcher.js itself, the words typed so far are searched,
// and this file stops listening.
(function () {
  var toggle = document.getElementById('search-toggle');
  var wrap = document.getElementById('search-wrapper');
  var bar = document.getElementById('searchbar');
  if (!toggle || !wrap || !bar) return;

  var FILES = ['elasticlunr.min.js', 'mark.min.js', 'searcher.js'];
  var SEARCH_KEY = 83; // 's', searcher.js's SEARCH_HOTKEY_KEYCODE
  var ESCAPE_KEY = 27;
  var started = false;
  var handedOver = false;
  var open = false; // the bar as this file last set it
  var ownChanges = 0; // aria-expanded writes of ours the observer has yet to see

  // book.js reads window.search.hasFocus() to leave the arrow keys to the
  // search bar; searcher.js keeps this object and replaces the function.
  window.search = window.search || {};
  if (!window.search.hasFocus) {
    window.search.hasFocus = function () {
      return document.activeElement === bar;
    };
  }

  function setOpen(yes) {
    open = yes;
    wrap.classList.toggle('hidden', !yes);
    ownChanges++;
    toggle.setAttribute('aria-expanded', yes ? 'true' : 'false');
    if (yes) {
      window.scrollTo(0, 0);
      bar.focus();
    }
  }

  function handOver() {
    handedOver = true;
    toggle.removeEventListener('click', onClick);
    document.removeEventListener('keydown', onKey);
    // searcher.js has just closed the bar (or opened it for ?search=);
    // reopen it if the reader had it open.
    if (!open || !wrap.classList.contains('hidden')) return;
    var typed = bar.value;
    toggle.click(); // searcher.js: open, scroll to the top, select the text
    bar.setSelectionRange(typed.length, typed.length);
    if (typed.trim() !== '') bar.dispatchEvent(new KeyboardEvent('keyup'));
  }

  function load() {
    if (started) return;
    started = true;
    new MutationObserver(function (records, observer) {
      for (var i = 0; i < records.length; i++) {
        if (ownChanges > 0) {
          ownChanges--;
          continue;
        }
        observer.disconnect();
        handOver();
        return;
      }
    }).observe(toggle, { attributes: true, attributeFilter: ['aria-expanded'] });
    (function next(i) {
      if (i === FILES.length) return;
      var s = document.createElement('script');
      s.src = path_to_root + FILES[i];
      s.onload = function () {
        next(i + 1);
      };
      document.head.appendChild(s);
    })(0);
  }

  function onClick() {
    if (handedOver) return;
    load();
    setOpen(!open);
  }

  function onKey(e) {
    if (handedOver || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
    var t = e.target;
    var typing = t && /^(?:input|select|textarea)$/i.test(t.nodeName);
    if (e.keyCode === ESCAPE_KEY && open) {
      e.preventDefault();
      setOpen(false);
      toggle.focus();
    } else if (e.keyCode === SEARCH_KEY && !typing) {
      e.preventDefault();
      load();
      setOpen(true);
    }
  }

  toggle.addEventListener('click', onClick);
  document.addEventListener('keydown', onKey);
  // A search result opens its page with ?highlight=, and a search survives a
  // reload as ?search=: searcher.js acts on both as soon as it starts.
  if (/[?&](?:search|highlight)=/.test(window.location.search)) load();
})();
