// server.js — обучение + бэкенд + встроенные примеры
const http = require('http');
const { URL } = require('url');
const fs = require('fs');

// === Нейросеть для генерации кода ===
class LanguageNeuralNetwork {
  constructor() {
    this.charToIndex = {};
    this.indexToChar = [];
    this.vocabSize = 0;
    this.seqLength = 40;
    this.learningRate = 0.01;
    this.weights = [];
    this.biases = [];
    this.hiddenSize = 100;
  }

  buildVocabulary(text) {
    const chars = [...new Set(text)];
    this.indexToChar = chars;
    this.charToIndex = chars.reduce((acc, char, i) => {
      acc[char] = i;
      return acc;
    }, {});
    this.vocabSize = chars.length;

    this.weights = Array(this.hiddenSize).fill(0).map(() => Array(this.vocabSize).fill(0).map(() => Math.random() * 0.1 - 0.05));
    this.biases = Array(this.vocabSize).fill(0).map(() => Math.random() * 0.1 - 0.05);
  }

  train(text) {
    if (!this.vocabSize) this.buildVocabulary(text);
    const chars = [...text];

    for (let i = 0; i < chars.length - this.seqLength; i++) {
      const sequence = chars.slice(i, i + this.seqLength);
      const target = chars[i + this.seqLength];

      if (this.charToIndex[target] === undefined) continue;

      const inputIndex = this.charToIndex[sequence[sequence.length - 1]];
      if (inputIndex === undefined) continue;

      const logits = this.weights.map(w => w[inputIndex] + this.biases[inputIndex]);
      const exps = logits.map(x => Math.exp(x));
      const sumExps = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map(x => x / sumExps);

      const targetIndex = this.charToIndex[target];
      const targetVec = Array(this.vocabSize).fill(0);
      targetVec[targetIndex] = 1;

      for (let j = 0; j < this.hiddenSize; j++) {
        for (let k = 0; k < this.vocabSize; k++) {
          this.weights[j][k] -= this.learningRate * (probs[k] - targetVec[k]) * (j === inputIndex ? 1 : 0);
        }
      }
      for (let k = 0; k < this.vocabSize; k++) {
        this.biases[k] -= this.learningRate * (probs[k] - targetVec[k]);
      }
    }
  }

  saveModel(filename) {
    const model = {
      charToIndex: this.charToIndex,
      indexToChar: this.indexToChar,
      vocabSize: this.vocabSize,
      seqLength: this.seqLength,
      weights: this.weights,
      biases: this.biases
    };
    fs.writeFileSync(filename, JSON.stringify(model));
    console.log(`Модель сохранена в ${filename}`);
  }

  loadModel(filename) {
    if (!fs.existsSync(filename)) return false;
    const model = JSON.parse(fs.readFileSync(filename, 'utf-8'));
    this.charToIndex = model.charToIndex;
    this.indexToChar = model.indexToChar;
    this.vocabSize = model.vocabSize;
    this.seqLength = model.seqLength;
    this.weights = model.weights;
    this.biases = model.biases;
    console.log(`Модель загружена из ${filename}`);
    return true;
  }

  generate(seed, length = 100) {
    if (!this.vocabSize) return 'Модель не обучена';
    let chars = [...seed];
    let result = [...chars];

    for (let i = 0; i < length; i++) {
      const lastChar = chars[chars.length - 1];
      const inputIndex = this.charToIndex[lastChar];
      if (inputIndex === undefined) break;

      const logits = this.weights.map(w => w[inputIndex] + this.biases[inputIndex]);
      const exps = logits.map(x => Math.exp(x));
      const sumExps = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map(x => x / sumExps);

      let rand = Math.random();
      let cumulative = 0;
      let nextIndex = 0;
      for (let j = 0; j < probs.length; j++) {
        cumulative += probs[j];
        if (rand < cumulative) {
          nextIndex = j;
          break;
        }
      }

      const nextChar = this.indexToChar[nextIndex];
      if (!nextChar) break;

      result.push(nextChar);
      chars.push(nextChar);
      chars = chars.slice(-this.seqLength);
    }

    return result.join('');
  }
}

// === Языки ===
const LANGUAGES = {
  js: 'javascript',
  py: 'python',
  java: 'java',
  cpp: 'cpp',
  c: 'c',
  html: 'html',
  css: 'css',
  ts: 'typescript',
  go: 'go',
  rust: 'rust'
};

// === Встроенные примеры кода ===
const CODE_SAMPLES = {
  javascript: `
function bubbleSort(arr) {
  const n = arr.length;
  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < n - i - 1; j++) {
      if (arr[j] > arr[j + 1]) {
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
      }
    }
  }
  return arr;
}

class Stack {
  constructor() {
    this.items = [];
  }

  push(element) {
    this.items.push(element);
  }

  pop() {
    if (this.items.length === 0) return "Underflow";
    return this.items.pop();
  }
}
`,
  python: `
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        return "Queue is empty"
`,
  java: `
public class Example {
    public static void main(String[] args) {
        int[] arr = {64, 34, 25, 12, 22, 11, 90};
        bubbleSort(arr);
        for (int i : arr) {
            System.out.print(i + " ");
        }
    }

    static void bubbleSort(int[] arr) {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0; j < n - i - 1; j++) {
                if (arr[j] > arr[j + 1]) {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }
}
`,
  cpp: `
#include <iostream>
using namespace std;

class Calculator {
public:
    int add(int a, int b) {
        return a + b;
    }

    int subtract(int a, int b) {
        return a - b;
    }
};

int main() {
    Calculator calc;
    cout << calc.add(5, 3) << endl;
    return 0;
}
`,
  c: `
#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int main() {
    printf("%d\\n", add(5, 3));
    return 0;
}
`,
  html: `
<!DOCTYPE html>
<html>
<head>
    <title>Example</title>
</head>
<body>
    <h1>Hello, World!</h1>
    <p>This is a sample HTML page.</p>
</body>
</html>
`,
  css: `
body {
    font-family: Arial, sans-serif;
    background-color: #f0f0f0;
}

h1 {
    color: #333;
}

p {
    font-size: 16px;
}
`,
  typescript: `
interface Stack<T> {
  push(item: T): void;
  pop(): T | undefined;
}

class GenericStack<T> implements Stack<T> {
  private items: T[] = [];

  push(item: T): void {
    this.items.push(item);
  }

  pop(): T | undefined {
    if (this.items.length === 0) return undefined;
    return this.items.pop();
  }
}
`,
  go: `
package main

import "fmt"

type Calculator struct{}

func (c Calculator) Add(a, b int) int {
    return a + b
}

func main() {
    calc := Calculator{}
    fmt.Println(calc.Add(5, 3))
}
`,
  rust: `
struct Calculator;

impl Calculator {
    fn add(&self, a: i32, b: i32) -> i32 {
        a + b
    }
}

fn main() {
    let calc = Calculator;
    println!("{}", calc.add(5, 3));
}
`
};

// === Загрузка или обучение моделей ===
const models = {};

for (const [ext, name] of Object.entries(LANGUAGES)) {
  const modelFile = `model_${name}.json`;
  let nn = new LanguageNeuralNetwork();

  if (nn.loadModel(modelFile)) {
    console.log(`${name} модель загружена.`);
  } else {
    console.log(`${name} модель не найдена. Обучаем...`);
    const codeText = CODE_SAMPLES[name] || '';

    if (codeText) {
      nn.train(codeText);
      nn.saveModel(modelFile);
      console.log(`${name} обучен.`);
    } else {
      console.log(`Нет кода для ${name}. Пропускаю обучение.`);
    }
  }

  models[name] = nn;
}

// === Нейросеть для текста ===
class TextGenerator {
  constructor() {
    this.wordToIndex = {};
    this.indexToWord = [];
    this.vocabSize = 0;
    this.seqLength = 2;
    this.learningRate = 0.01;
    this.weights = [];
    this.biases = [];
    this.hiddenSize = 30;
  }

  buildVocabulary(text) {
    const words = text.split(/\s+/);
    const uniqueWords = [...new Set(words)];
    this.indexToWord = uniqueWords;
    this.wordToIndex = uniqueWords.reduce((acc, word, i) => {
      acc[word] = i;
      return acc;
    }, {});
    this.vocabSize = uniqueWords.length;

    this.weights = Array(this.hiddenSize).fill(0).map(() => Array(this.vocabSize).fill(0).map(() => Math.random() * 0.1 - 0.05));
    this.biases = Array(this.vocabSize).fill(0).map(() => Math.random() * 0.1 - 0.05);
  }

  train(text) {
    if (!this.vocabSize) this.buildVocabulary(text);
    const words = text.split(/\s+/);
    if (words.length <= this.seqLength) return;

    for (let i = 0; i < words.length - this.seqLength; i++) {
      const sequence = words.slice(i, i + this.seqLength);
      const target = words[i + this.seqLength];

      if (this.wordToIndex[target] === undefined) continue;

      const inputIndex = this.wordToIndex[sequence[this.seqLength - 1]];
      if (inputIndex === undefined) continue;

      const logits = this.weights.map(w => w[inputIndex] + this.biases[inputIndex]);
      const exps = logits.map(x => Math.exp(x));
      const sumExps = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map(x => x / sumExps);

      const targetIndex = this.wordToIndex[target];
      const targetVec = Array(this.vocabSize).fill(0);
      targetVec[targetIndex] = 1;

      for (let j = 0; j < this.hiddenSize; j++) {
        for (let k = 0; k < this.vocabSize; k++) {
          this.weights[j][k] -= this.learningRate * (probs[k] - targetVec[k]) * (j === inputIndex ? 1 : 0);
        }
      }
      for (let k = 0; k < this.vocabSize; k++) {
        this.biases[k] -= this.learningRate * (probs[k] - targetVec[k]);
      }
    }
  }

  generate(seed, length = 5) {
    if (!this.vocabSize) return 'Словарь пуст';
    let words = seed.split(/\s+/).slice(-this.seqLength);
    let result = [...words];

    for (let i = 0; i < length; i++) {
      const lastWord = words[words.length - 1];
      const inputIndex = this.wordToIndex[lastWord];
      if (inputIndex === undefined) break;

      const logits = this.weights.map(w => w[inputIndex] + this.biases[inputIndex]);
      const exps = logits.map(x => Math.exp(x));
      const sumExps = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map(x => x / sumExps);

      let rand = Math.random();
      let cumulative = 0;
      let nextIndex = 0;
      for (let j = 0; j < probs.length; j++) {
        cumulative += probs[j];
        if (rand < cumulative) {
          nextIndex = j;
          break;
        }
      }

      const nextWord = this.indexToWord[nextIndex];
      if (!nextWord) break;

      result.push(nextWord);
      words.push(nextWord);
      words = words.slice(-this.seqLength);
    }

    return result.join(' ');
  }
}

// === Обучение текстовой нейросети ===
const textGen = new TextGenerator();
const trainingText = `
hello world
how are you
i am fine
what is your name
i am a bot
can you help me
yes i can
do you like cats
cats are cute
do you like dogs
dogs are loyal
the sun is bright
the moon is dark
`;
textGen.train(trainingText);

// === Определение языка ===
function detectLanguage(message) {
  message = message.toLowerCase();
  if (message.includes('javascript') || message.includes('js')) return 'javascript';
  if (message.includes('python') || message.includes('py')) return 'python';
  if (message.includes('java')) return 'java';
  if (message.includes('c++') || message.includes('cpp')) return 'cpp';
  if (message.includes('c') && !message.includes('cpp') && !message.includes('css')) return 'c';
  if (message.includes('html')) return 'html';
  if (message.includes('css')) return 'css';
  if (message.includes('typescript') || message.includes('ts')) return 'typescript';
  if (message.includes('go') || message.includes('golang')) return 'go';
  if (message.includes('rust')) return 'rust';
  return null;
}

// === Чат-бот ===
class ChatBot {
  constructor() {
    this.history = [];
    this.textGen = textGen;
  }

  async processMessage(message) {
    this.history.push({ user: message });

    const lang = detectLanguage(message);
    if (lang && models[lang]) {
      const code = models[lang].generate(message, 150);
      this.history.push({ bot: code });
      return { type: 'code', content: code };
    }

    if (message.includes('код') || message.includes('function') || message.includes('class')) {
      const code = models['javascript'] ? models['javascript'].generate(message, 150) : 'Язык не обучен.';
      this.history.push({ bot: code });
      return { type: 'code', content: code };
    }

    if (message.includes('картинка') || message.includes('изображение') || message.includes('image')) {
      return { type: 'image', content: 'Генерация изображений невозможна без внешних API или библиотек.' };
    }

    const response = this.textGen.generate(message, 8);
    this.history.push({ bot: response });
    return { type: 'text', content: response };
  }
}

const bot = new ChatBot();

// === Обработчик запросов ===
const server = http.createServer(async (req, res) => {
  const reqUrl = new URL(req.url, `http://${req.headers.host}`);

  // === Отдаём HTML-фронтенд по корню ===
  if (req.method === 'GET' && reqUrl.pathname === '/') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(`
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <title>Чат-бот с генерацией</title>
  <style>
    body { font-family: sans-serif; max-width: 700px; margin: 40px auto; padding: 20px; }
    #chatLog { height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; background: #f9f9f9; }
    .user { text-align: right; color: #007bff; }
    .bot { text-align: left; color: #28a745; }
    .code { background: #f4f4f4; padding: 10px; border-left: 3px solid #007bff; white-space: pre-wrap; font-family: monospace; }
    textarea { width: 100%; height: 60px; margin: 10px 0; }
    button { padding: 10px 20px; font-size: 16px; }
  </style>
</head>
<body>
  <h2>Чат-бот (текст, код)</h2>
  <div id="chatLog"></div>
  <textarea id="message" placeholder="Введите сообщение..."></textarea><br>
  <button onclick="send()">Отправить</button>

  <script>
    const chatLog = document.getElementById('chatLog');

    function addMessage(user, text, type = 'text') {
      const div = document.createElement('div');
      div.className = user;
      if (type === 'code') {
        const pre = document.createElement('div');
        pre.className = 'code';
        pre.textContent = text;
        div.appendChild(pre);
      } else {
        div.textContent = text;
      }
      chatLog.appendChild(div);
      chatLog.scrollTop = chatLog.scrollHeight;
    }

    async function send() {
      const input = document.getElementById('message');
      const message = input.value.trim();
      if (!message) return;

      addMessage('user', message);
      input.value = '';

      const response = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
      });

      const data = await response.json();

      if (response.ok) {
        addMessage('bot', data.content, data.type);
      } else {
        addMessage('bot', 'Ошибка: ' + (data.error || 'Неизвестная ошибка'));
      }
    }

    document.getElementById('message').addEventListener('keypress', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        send();
      }
    });
  </script>
</body>
</html>
    `);
    return;
  }

  // === API ===
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  if (req.method === 'POST' && reqUrl.pathname === '/chat') {
    let body = '';
    req.on('data', chunk => body += chunk.toString());
    req.on('end', async () => {
      try {
        const data = JSON.parse(body);
        const message = data.message;

        if (typeof message !== 'string') {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Message должен быть строкой' }));
          return;
        }

        const response = await bot.processMessage(message);

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(response));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Ошибка сервера' }));
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