const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

// Импортируем trainingData из отдельного файла
const trainingDataArray = require('./trainingData.js'); // Путь к файлу с данными

// --- Нейросеть ---
function createMarkovChain(text) {
    const words = text.split(' ');
    const chain = {};

    for (let i = 0; i < words.length - 1; i++) {
        const currentWord = words[i];
        const nextWord = words[i + 1];

        if (!chain[currentWord]) {
            chain[currentWord] = [];
        }
        chain[currentWord].push(nextWord);
    }

    return chain;
}

function generateText(markovChain, seed, length = 20) {
    const words = seed.split(' ');
    let currentWord = words[words.length - 1];

    for (let i = 0; i < length; i++) {
        const nextWords = markovChain[currentWord];
        if (!nextWords || nextWords.length === 0) {
            break; // Если следующее слово невозможно, останавливаем генерацию
        }
        const randomIndex = Math.floor(Math.random() * nextWords.length);
        const nextWord = nextWords[randomIndex];
        words.push(nextWord);
        currentWord = nextWord;
    }

    return words.join(' ');
}

// Объединяем все строки из trainingDataArray в одну строку
const allText = trainingDataArray.join(' ').replace(/\s+/g, ' ').trim();

// Создаем цепочку Маркова из обучающих данных
const markovChain = createMarkovChain(allText);

// --- Веб-сервер ---
const server = http.createServer((req, res) => {
    const parsedUrl = url.parse(req.url, true);
    const pathname = parsedUrl.pathname;

    // Настройка CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    // Обработка OPTIONS запросов для CORS
    if (req.method === 'OPTIONS') {
        res.writeHead(204); // No Content
        res.end();
        return;
    }

    if (pathname === '/' && req.method === 'GET') {
        // Отправляем frontend.html
        const filePath = path.join(__dirname, 'frontend.html');
        fs.readFile(filePath, (err, content) => {
            if (err) {
                res.writeHead(500);
                res.end('Error loading frontend.');
                console.error("Error reading frontend.html:", err);
            } else {
                res.writeHead(200, { 'Content-Type': 'text/html' });
                res.end(content);
            }
        });
    } else if (pathname === '/generate' && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });
        req.on('end', () => {
            try {
                const { input } = JSON.parse(body);
                if (typeof input !== 'string' || input.trim() === '') {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Invalid input: expected a non-empty string.' }));
                    return;
                }

                // Генерируем текст
                const generated = generateText(markovChain, input.trim(), 50); // Длина 50 слов

                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ output: generated }));
            } catch (error) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Invalid JSON or missing input field.' }));
                console.error("Error processing generate request:", error);
            }
        });
    } else {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Not Found' }));
    }
});

// Используем переменную окружения PORT или 10000 по умолчанию
const PORT = process.env.PORT || 10000;
server.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Training data loaded: ${trainingDataArray.length} entries`);
});