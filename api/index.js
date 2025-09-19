// api/index.js
const express = require('express');
const app = express();


app.use(express.json({ limit: '1mb' })); // sesuaikan limit; jangan set terlalu besar di function


// contoh GET
app.get('/hello', (req, res) => {
res.json({ ok: true, message: 'Hello from Vercel API' });
});


// contoh POST ringan (metadata atau URL dari presigned upload)
app.post('/predict', async (req, res) => {
try {
const { fileUrl, metadata } = req.body;
if (!fileUrl) return res.status(400).json({ error: 'fileUrl missing' });


// Jangan mengunduh file besar dari sini; idealnya client upload ke S3/Storage
// dan kamu hanya menerima fileUrl untuk pemrosesan ringan atau job enqueue.


// contoh respons sederhana
return res.json({ ok: true, received: { fileUrl, metadata } });
} catch (err) {
console.error(err);
return res.status(500).json({ error: 'internal error' });
}
});


module.exports = app;