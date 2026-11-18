// Where the PathSnap API lives.
//
// ""  (default)            same origin. The API serves this page (uvicorn at
//                          http://127.0.0.1:8000/ui/ or behind a Cloudflare Tunnel),
//                          so relative /api/... calls just work.
// "http://127.0.0.1:8000"  the API is on another laptop/port and this page is served
//                          separately, e.g. python -m http.server 5500 --directory frontend
//
// Never commit a tunnel URL or a LAN IP here; use "" for the tunnel.
window.PATHSNAP_API = "";
