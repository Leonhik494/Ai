// server.js
const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

class SimpleNeuralNetwork {
    constructor(vocabSize, hiddenSize = 50) {
        this.vocabSize = vocabSize;
        this.hiddenSize = hiddenSize;
        
        // Initialize weights randomly
        this.Wxh = this.randomMatrix(hiddenSize, vocabSize, 0.1);
        this.Whh = this.randomMatrix(hiddenSize, hiddenSize, 0.1);
        this.Why = this.randomMatrix(vocabSize, hiddenSize, 0.1);
        this.bh = new Array(hiddenSize).fill(0);
        this.by = new Array(vocabSize).fill(0);
    }

    randomMatrix(rows, cols, scale) {
        const matrix = [];
        for (let i = 0; i < rows; i++) {
            const row = [];
            for (let j = 0; j < cols; j++) {
                row.push((Math.random() * 2 - 1) * scale);
            }
            matrix.push(row);
        }
        return matrix;
    }

    matrixVectorMultiply(matrix, vector) {
        const result = new Array(matrix.length).fill(0);
        for (let i = 0; i < matrix.length; i++) {
            let sum = 0;
            for (let j = 0; j < matrix[i].length; j++) {
                sum += matrix[i][j] * vector[j];
            }
            result[i] = sum;
        }
        return result;
    }

    vectorAdd(v1, v2) {
        const result = new Array(v1.length);
        for (let i = 0; i < v1.length; i++) {
            result[i] = v1[i] + v2[i];
        }
        return result;
    }

    vectorTanh(v) {
        const result = new Array(v.length);
        for (let i = 0; i < v.length; i++) {
            result[i] = Math.tanh(v[i]);
        }
        return result;
    }

    softmax(v) {
        const max = Math.max(...v);
        const exps = v.map(x => Math.exp(x - max));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => x / sum);
    }

    forward(inputs) {
        const hiddens = [];
        const outputs = [];
        let h = new Array(this.hiddenSize).fill(0);

        for (const input of inputs) {
            const x = this.oneHot(input, this.vocabSize);
            const hRaw = this.vectorAdd(
                this.matrixVectorMultiply(this.Wxh, x),
                this.matrixVectorMultiply(this.Whh, h)
            );
            h = this.vectorTanh(this.vectorAdd(hRaw, this.bh));
            
            const oRaw = this.vectorAdd(
                this.matrixVectorMultiply(this.Why, h),
                this.by
            );
            const o = this.softmax(oRaw);
            
            hiddens.push(h);
            outputs.push(o);
        }

        return { hiddens, outputs };
    }

    oneHot(index, size) {
        const result = new Array(size).fill(0);
        result[index] = 1;
        return result;
    }

    train(text, epochs = 100, learningRate = 0.01) {
        const { vocab, charToIdx, idxToChar } = this.createVocabulary(text);
        this.charToIdx = charToIdx;
        this.idxToChar = idxToChar;

        const inputs = text.split('').map(char => charToIdx[char]);
        const targets = inputs.slice(1).concat([inputs[0]]);

        for (let epoch = 0; epoch < epochs; epoch++) {
            const { hiddens, outputs } = this.forward(inputs);
            
            // Calculate loss and gradients
            let loss = 0;
            for (let t = 0; t < inputs.length; t++) {
                const target = targets[t];
                loss += -Math.log(outputs[t][target] + 1e-9);
                
                // Simple gradient calculation (not full backprop for simplicity)
                outputs[t][target] -= 1;
            }
            
            // Update weights using gradients (simplified)
            for (let t = inputs.length - 1; t >= 0; t--) {
                const dy = outputs[t];
                const h = hiddens[t];
                
                // Update Why
                for (let i = 0; i < this.Why.length; i++) {
                    for (let j = 0; j < this.Why[i].length; j++) {
                        this.Why[i][j] -= learningRate * dy[i] * h[j];
                    }
                }
                
                // Update by
                for (let i = 0; i < this.by.length; i++) {
                    this.by[i] -= learningRate * dy[i];
                }
            }
        }
    }

    createVocabulary(text) {
        const chars = [...new Set(text)];
        const charToIdx = {};
        const idxToChar = {};
        
        chars.forEach((char, idx) => {
            charToIdx[char] = idx;
            idxToChar[idx] = char;
        });
        
        return { vocab: chars, charToIdx, idxToChar };
    }

    predict(seed, length = 100) {
        if (!this.charToIdx) return '';
        
        const inputs = seed.split('').map(char => this.charToIdx[char]).filter(idx => idx !== undefined);
        if (inputs.length === 0) return '';
        
        let h = new Array(this.hiddenSize).fill(0);
        let generated = seed;
        
        for (let i = 0; i < length; i++) {
            const x = this.oneHot(inputs[inputs.length - 1], this.vocabSize);
            const hRaw = this.vectorAdd(
                this.matrixVectorMultiply(this.Wxh, x),
                this.matrixVectorMultiply(this.Whh, h)
            );
            h = this.vectorTanh(this.vectorAdd(hRaw, this.bh));
            
            const oRaw = this.vectorAdd(
                this.matrixVectorMultiply(this.Why, h),
                this.by
            );
            const o = this.softmax(oRaw);
            
            // Sample next character
            const r = Math.random();
            let cumulative = 0;
            let nextCharIdx = 0;
            for (let j = 0; j < o.length; j++) {
                cumulative += o[j];
                if (r < cumulative) {
                    nextCharIdx = j;
                    break;
                }
            }
            
            generated += this.idxToChar[nextCharIdx];
            inputs.push(nextCharIdx);
        }
        
        return generated;
    }

    saveWeights(filename) {
        const weights = {
            Wxh: this.Wxh,
            Whh: this.Whh,
            Why: this.Why,
            bh: this.bh,
            by: this.by,
            charToIdx: this.charToIdx,
            idxToChar: this.idxToChar,
            vocabSize: this.vocabSize,
            hiddenSize: this.hiddenSize
        };
        fs.writeFileSync(filename, JSON.stringify(weights));
    }

    loadWeights(filename) {
        const weights = JSON.parse(fs.readFileSync(filename, 'utf8'));
        this.Wxh = weights.Wxh;
        this.Whh = weights.Whh;
        this.Why = weights.Why;
        this.bh = weights.bh;
        this.by = weights.by;
        this.charToIdx = weights.charToIdx;
        this.idxToChar = weights.idxToChar;
        this.vocabSize = weights.vocabSize;
        this.hiddenSize = weights.hiddenSize;
    }
}

const sampleText = `The quick brown fox jumps over the lazy dog. 
Artificial intelligence is transforming the world. 
Machine learning algorithms can learn from data.
Neural networks are inspired by the human brain.
JavaScript is a versatile programming language.
Node.js allows running JavaScript on the server.
HTTP servers handle requests and responses.
JSON is a lightweight data interchange format.
Programming requires logic and creativity.
Technology advances rapidly every year.
The future holds many possibilities.
Innovation drives progress in society.
Computers process information quickly.
Algorithms solve problems efficiently.
Data structures organize information.
Coding is a valuable skill.
Software development involves many disciplines.
Debugging is an important part of programming.
Version control helps manage code changes.
Collaboration improves software quality.`;

const nn = new SimpleNeuralNetwork(100); // Will be adjusted dynamically

// Train the network
console.log('Training neural network...');
nn.train(sampleText, 50, 0.01);
console.log('Training completed.');

const server = http.createServer((req, res) => {
    const parsedUrl = url.parse(req.url, true);
    const pathname = parsedUrl.pathname;

    // Handle CORS
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
        res.writeHead(200);
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
                const input = data.input || '';
                
                if (!nn.charToIdx) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: 'Model not trained' }));
                    return;
                }
                
                const output = nn.predict(input, 100);
                
                res.writeHead(200, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ output }));
            } catch (error) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'Invalid JSON' }));
            }
        });
    } else if (pathname === '/frontend.html' && req.method === 'GET') {
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
    console.log(`Server running on port ${PORT}`);
});