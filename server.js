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
    // Разбиваем семя на слова
    const seedWords = seed.split(' ').filter(word => word.trim() !== '');
    if (seedWords.length === 0) {
        // Если семя пустое, начать с случайного слова из цепочки
        const startWords = Object.keys(markovChain);
        if (startWords.length === 0) return "Training data is empty.";
        let currentWord = startWords[Math.floor(Math.random() * startWords.length)];
        const words = [currentWord];

        for (let i = 0; i < length - 1; i++) { // -1 потому что первое слово уже добавлено
            const nextWords = markovChain[currentWord];
            if (!nextWords || nextWords.length === 0) {
                 // Если нет следующего слова, останавливаем генерацию
                 break;
            }
            const randomIndex = Math.floor(Math.random() * nextWords.length);
            const nextWord = nextWords[randomIndex];
            words.push(nextWord);
            currentWord = nextWord;
        }
        return words.join(' ');
    } else {
        // Используем последнее слово из семени как начальное состояние
        // (или последнее слово, если в семени несколько)
        let currentWord = seedWords[seedWords.length - 1]; // Берем последнее слово из семени

        // Начинаем генерацию, но не включаем само семя в начало результата
        // Вместо этого сразу ищем следующее слово после currentWord
        const nextWords = markovChain[currentWord];
        if (!nextWords || nextWords.length === 0) {
             // Если после seed-слова в цепочке нет продолжения, возвращаем семя или сообщение
             // return seed; // Это приведет к повторению
             // Лучше попробовать начать с другого слова из семени или случайного
             // Но для простоты, если нет продолжения после последнего слова, пробуем с первого
             const firstSeedWord = seedWords[0];
             if (markovChain[firstSeedWord] && markovChain[firstSeedWord].length > 0) {
                 currentWord = firstSeedWord;
             } else {
                 // Если и с первым словом нет продолжения, возвращаем сообщение
                 return "Не могу продолжить фразу '" + seed + "'. Попробуйте другое.";
             }
        }

        const generatedWords = [];
        let attempts = 0; // Ограничим количество попыток, чтобы избежать бесконечного цикла
        const maxAttempts = length * 2; // Допустим, в 2 раза больше длины

        while (generatedWords.length < length && attempts < maxAttempts) {
            const nextWords = markovChain[currentWord];
            if (!nextWords || nextWords.length === 0) {
                 break; // Если нет следующего слова, останавливаем генерацию
            }
            const randomIndex = Math.floor(Math.random() * nextWords.length);
            const nextWord = nextWords[randomIndex];
            generatedWords.push(nextWord);
            currentWord = nextWord;
            attempts++;
        }

        // Возвращаем сгенерированный текст, НЕ включая исходное семя
        if (generatedWords.length > 0) {
            return generatedWords.join(' ');
        } else {
            // Если не удалось сгенерировать ничего нового
            return "Попробуйте более распространённое слово или фразу.";
        }
    }
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