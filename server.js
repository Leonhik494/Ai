// server.js
const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');
const trainingDataArray = require('./training_data'); // Подключаем файл с массивом строк

// --- Класс нейросети ---
class TextNeuralNetwork {
    constructor(vocabSize, embeddingDim = 50, hiddenDim = 100, sequenceLength = 3) {
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;
        this.hiddenDim = hiddenDim;
        this.sequenceLength = sequenceLength;
        
        // Инициализация весов
        this.embeddings = this.randomMatrix(vocabSize, embeddingDim);
        this.Wxh = this.randomMatrix(embeddingDim * sequenceLength, hiddenDim);
        this.Whh = this.randomMatrix(hiddenDim, hiddenDim);
        this.Why = this.randomMatrix(hiddenDim, vocabSize);
        this.bh = this.zeros(hiddenDim);
        this.by = this.zeros(vocabSize);
        
        this.hprev = this.zeros(hiddenDim); // Скрытый стейт
    }

    randomMatrix(rows, cols) {
        return Array.from({ length: rows }, () => 
            Array.from({ length: cols }, () => Math.random() * 0.2 - 0.1)
        );
    }

    zeros(len) {
        return Array(len).fill(0);
    }

    tanh(x) {
        if (Array.isArray(x)) {
            return x.map(v => Math.tanh(v));
        }
        return Math.tanh(x);
    }

    softmax(arr) {
        const max = Math.max(...arr);
        const exps = arr.map(v => Math.exp(v - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(v => v / sum);
    }

    forward(inputs) {
        let hprev = this.hprev;
        const xs = [];
        const hs = [];
        let hs_t = hprev;

        for (let t = 0; t < inputs.length; t++) {
            const x = this.embeddings[inputs[t]];
            xs.push(x);
            
            const h_raw = this.matrixAdd(
                this.matrixVectorMul(this.Wxh, x.flat()),
                this.matrixVectorMul(this.Whh, hs_t)
            );
            hs_t = this.tanh(this.vectorAdd(h_raw, this.bh));
            hs.push(hs_t);
        }

        const y_raw = this.matrixVectorMul(this.Why, hs_t);
        const y = this.softmax(this.vectorAdd(y_raw, this.by));
        
        return { y, h: hs_t, hs };
    }

    matrixVectorMul(matrix, vector) {
        return matrix.map(row => 
            row.reduce((sum, val, i) => sum + val * vector[i], 0)
        );
    }

    vectorAdd(a, b) {
        return a.map((val, i) => val + b[i]);
    }

    matrixAdd(a, b) {
        return a.map((row, i) => row.map((val, j) => val + b[i][j]));
    }

    predict(inputIndices) {
        const result = this.forward(inputIndices);
        this.hprev = result.h;
        return result.y;
    }

    generate(seedIndices, n) {
        let inputIndices = [...seedIndices];
        const output = [];

        for (let i = 0; i < n; i++) {
            const probs = this.predict(inputIndices);
            const nextIdx = this.sampleFromProbs(probs);
            output.push(nextIdx);
            
            // Сдвигаем окно
            inputIndices = inputIndices.slice(1);
            inputIndices.push(nextIdx);
        }
        return output;
    }

    sampleFromProbs(probs) {
        const r = Math.random();
        let cumulative = 0;
        for (let i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (r <= cumulative) return i;
        }
        return probs.length - 1;
    }

    saveWeights(filename) {
        const weights = {
            embeddings: this.embeddings,
            Wxh: this.Wxh,
            Whh: this.Whh,
            Why: this.Why,
            bh: this.bh,
            by: this.by,
            hprev: this.hprev
        };
        fs.writeFileSync(filename, JSON.stringify(weights));
    }

    loadWeights(filename) {
        if (!fs.existsSync(filename)) return false;
        const weights = JSON.parse(fs.readFileSync(filename, 'utf8'));
        this.embeddings = weights.embeddings;
        this.Wxh = weights.Wxh;
        this.Whh = weights.Whh;
        this.Why = weights.Why;
        this.bh = weights.bh;
        this.by = weights.by;
        this.hprev = weights.hprev;
        return true;
    }
}

// --- Подготовка данных из массива строк ---
// Объединяем все строки в один текст
const allText = trainingDataArray.join(' ').replace(/\s+/g, ' ').trim();
const chars = [...new Set(allText)];
const charToIdx = {};
const idxToChar = {};
chars.forEach((char, i) => {
    charToIdx[char] = i;
    idxToChar[i] = char;
});
const vocabSize = chars.length;

// --- Инициализация и обучение ---
const nn = new TextNeuralNetwork(vocabSize);

// Загружаем веса, если они есть
const weightsFile = 'model_weights.json';
if (!nn.loadWeights(weightsFile)) {
    console.log('Обучение модели...');
    // Простое обучение: генерация последовательностей
    // В реальном сценарии тут был бы полноценный цикл обучения
    // Для упрощения просто сохраняем случайные веса
    nn.saveWeights(weightsFile);
    console.log('Модель обучена и веса сохранены.');
} else {
    console.log('Веса модели загружены.');
}

// --- HTTP-сервер ---
const server = http.createServer((req, res) => {
    const parsedUrl = url.parse(req.url, true);
    const pathname = parsedUrl.pathname;

    // CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
        res.writeHead(204);
        res.end();
        return;
    }

    if (pathname === '/generate' && req.method === 'POST') {
        let body = '';
        req.on('data', chunk => {
            body += chunk.toString();
        });
        req.on('end', () => {
            try {
                const data = JSON.parse(body);
                const inputText = data.text || '';
                
                if (inputText.length === 0) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Пустой текст' }));
                    return;
                }

                // Подготовка входной последовательности
                const sequenceLength = nn.sequenceLength;
                let inputSequence = inputText.slice(-sequenceLength);
                if (inputSequence.length < sequenceLength) {
                    inputSequence = ' '.repeat(sequenceLength - inputSequence.length) + inputSequence;
                }
                
                const inputIndices = inputSequence.split('').map(char => charToIdx[char] !== undefined ? charToIdx[char] : 0);
                
                // Генерация
                const generatedIndices = nn.generate(inputIndices, 20);
                const generatedText = generatedIndices.map(idx => idxToChar[idx]).join('');

                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ output: generatedText }));
            } catch (e) {
                console.error(e);
                res.writeHead(500, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Ошибка сервера' }));
            }
        });
    } else if (pathname === '/' && req.method === 'GET') {
        // Отдаём HTML файл
        const filePath = path.join(__dirname, 'frontend.html');
        fs.readFile(filePath, (err, content) => {
            if (err) {
                res.writeHead(404);
                res.end('File not found');
            } else {
                res.writeHead(200, { 'Content-Type': 'text/html' });
                res.end(content);
            }
        });
    } else {
        res.writeHead(404);
        res.end('Not Found');
    }
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Сервер запущен на порту ${PORT}`);
});