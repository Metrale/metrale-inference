// Keyboard access to code blocks and tables that scroll sideways (WCAG 2.1.1).
//
// A wide code block (mdBook's scroller is the <code> inside the <pre>) or table
// wrapper scrolls with a pointer but cannot take focus, so a keyboard user
// cannot read the rest of the line. Each one that overflows becomes a named,
// focusable region, named by its kind and its place on the page (landmark names
// must be unique); arrow keys then scroll it, and metrale.css's :focus-visible
// draws the copper ring. One that fits is left out of the tab order, and the
// set is re-measured when the width changes. mdBook offers no build-time hook
// for this, so it runs in the page, once parsed (after book.js, which index.hbs
// defers).
//
// book.js turns Left and Right on the document into previous and next chapter.
// Inside a region those keys must scroll it instead, so they stop there.
(function () {
  var KINDS = [
    ['.content pre > code', 'Code block'],
    ['.content .table-wrapper', 'Table'],
  ];
  function mark() {
    for (var k = 0; k < KINDS.length; k++) {
      var els = document.querySelectorAll(KINDS[k][0]);
      for (var i = 0; i < els.length; i++) {
        var el = els[i];
        if (el.scrollWidth > el.clientWidth + 1) {
          el.setAttribute('tabindex', '0');
          el.setAttribute('role', 'region');
          el.setAttribute('aria-label', KINDS[k][1] + ' ' + (i + 1) + ', scrolls sideways');
        } else if (el.getAttribute('role') === 'region') {
          el.removeAttribute('tabindex');
          el.removeAttribute('role');
          el.removeAttribute('aria-label');
        }
      }
    }
  }
  document.addEventListener(
    'keydown',
    function (e) {
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      var t = e.target;
      if (t && t.getAttribute && t.getAttribute('role') === 'region' && t.closest('.content')) e.stopPropagation();
    },
    true,
  );
  var pending = 0;
  window.addEventListener('resize', function () {
    if (pending) cancelAnimationFrame(pending);
    pending = requestAnimationFrame(function () {
      pending = 0;
      mark();
    });
  });
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mark);
  else mark();
  // Manrope and Plex Mono arrive after first paint and change line widths.
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(mark);
})();
