(function () {
  var path = window.location.pathname;
  var hasExtension = /\/[^/]+\.[^/]+$/.test(path);
  if (hasExtension || path.endsWith("/")) {
    return;
  }
  if (path === "/turboagents" || path.endsWith("/turboagents")) {
    window.location.replace(path + "/" + window.location.search + window.location.hash);
  }
})();
