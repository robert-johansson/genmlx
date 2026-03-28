// Local HTTPS server for GenStudio development
// Run: node serve.mjs
// Then open https://<your-mac-ip>:8443/studio/ on iPad
import { createServer } from 'https';
import { readFileSync, existsSync, statSync } from 'fs';
import { join, extname } from 'path';
import { networkInterfaces } from 'os';

const PORT = 8443;
const ROOT = join(import.meta.dirname, 'docs');

const MIME = {
  '.html': 'text/html',
  '.js':   'application/javascript',
  '.mjs':  'application/javascript',
  '.css':  'text/css',
  '.json': 'application/json',
  '.cljs': 'text/plain',
  '.png':  'image/png',
  '.svg':  'image/svg+xml',
};

const server = createServer(
  { key: readFileSync('key.pem'), cert: readFileSync('cert.pem') },
  (req, res) => {
    let path = join(ROOT, req.url === '/' ? '/index.html' : req.url);
    if (existsSync(path) && statSync(path).isDirectory()) {
      path = join(path, 'index.html');
    }
    if (!existsSync(path)) {
      res.writeHead(404);
      res.end('Not found');
      return;
    }
    const mime = MIME[extname(path)] || 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': mime });
    res.end(readFileSync(path));
  }
);

server.listen(PORT, '0.0.0.0', () => {
  console.log(`\nGenStudio HTTPS server running:\n`);
  console.log(`  Local:   https://localhost:${PORT}/studio/`);

  // Show LAN addresses for iPad access
  const nets = networkInterfaces();
  for (const name of Object.keys(nets)) {
    for (const net of nets[name]) {
      if (net.family === 'IPv4' && !net.internal) {
        console.log(`  iPad:    https://${net.address}:${PORT}/studio/`);
      }
    }
  }
  console.log(`\nNote: iPad will show a certificate warning — tap "Show Details" → "visit this website"\n`);
});
