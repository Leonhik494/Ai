// server.js — обучение + бэкенд + встроенные примеры + перевод русского на английский и обратно
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

    const generated = result.join('');
    if (generated.startsWith(seed) && generated.length === seed.length) {
      return generated + ' // (не удалось сгенерировать продолжение)';
    }
    return generated;
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

// === Словарь для перевода русских фраз в английские ===
const ruToEn = {
  'привет': 'hello',
  'мир': 'world',
  'как': 'how',
  'дела': 'are you',
  'у': 'you',
  'я': 'i',
  'хорошо': 'fine',
  'что': 'what',
  'твоё': 'your',
  'имя': 'name',
  'бот': 'bot',
  'ты': 'you',
  'могу': 'can',
  'помочь': 'help',
  'да': 'yes',
  'нет': 'no',
  'люблю': 'like',
  'коты': 'cats',
  'собаки': 'dogs',
  'солнце': 'sun',
  'луну': 'moon',
  'яркое': 'bright',
  'тёмная': 'dark',
  'птицы': 'birds',
  'летят': 'fly',
  'высоко': 'high',
  'рыбы': 'fish',
  'плавают': 'swim',
  'глубоко': 'deep',
  'земля': 'world',
  'круглая': 'round',
  'небо': 'sky',
  'синее': 'blue',
  'код': 'code',
  'писать': 'write',
  'функция': 'function',
  'класс': 'class',
  'объект': 'object',
  'метод': 'method',
  'делать': 'do',
  'вещи': 'things',
  'реальные': 'real',
  'фейк': 'fake',
  'понимаю': 'understand',
  'учусь': 'learn',
  'думаю': 'think',
  'говорю': 'talk',
  'слушаю': 'hear',
  'вижу': 'see',
  'чувствую': 'feel',
  'знаю': 'know',
  'ничего': 'nothing',
  'что-то': 'something',
  'всё': 'everything',
  'возможно': 'possible',
  'невозможно': 'impossible',
  'умный': 'smart',
  'добрый': 'kind',
  'полезный': 'helpful',
  'друг': 'friend',
  'добрый': 'good',
  'плохой': 'bad',
  'лучше': 'better',
  'лучший': 'best',
  'худший': 'worst',
  'спасибо': 'thank you',
  'пожалуйста': 'you are welcome',
  'хорошего': 'have a nice',
  'дня': 'day',
  'пока': 'goodbye',
  'увидимся': 'see you later',
  'позже': 'later',
  'время': 'time',
  'драгоценное': 'precious',
  'ценное': 'valuable',
  'важное': 'important',
  'значимое': 'significant',
  'осмысленное': 'meaningful',
  'глубокое': 'deep',
  'глубокомысленное': 'profound',
  'мудрое': 'wise',
  'интеллектуальный': 'intelligent',
  'ловкий': 'clever',
  'быстрый': 'quick',
  'медленный': 'slow',
  'вверх': 'up',
  'вниз': 'down',
  'влево': 'left',
  'вправо': 'right',
  'мне': 'me',
  'нам': 'us',
  'вам': 'you',
  'им': 'them',
  'её': 'her',
  'его': 'his',
  'их': 'their',
  'моё': 'my',
  'наше': 'our',
  'ваше': 'your',
  'себя': 'myself',
  'все': 'all',
  'вся': 'all',
  'также': 'also',
  'ещё': 'also',
  'еще': 'also',
  'тоже': 'too',
  'всегда': 'always',
  'никогда': 'never',
  'иногда': 'sometimes',
  'часто': 'often',
  'редко': 'rarely',
  'сейчас': 'now',
  'тогда': 'then',
  'здесь': 'here',
  'там': 'there',
  'почему': 'why',
  'когда': 'when',
  'где': 'where',
  'кто': 'who',
  'какой': 'which',
  'какая': 'which',
  'какое': 'which',
  'какие': 'which',
  'какого': 'which',
  'какой-то': 'some',
  'некоторые': 'some',
  'много': 'many',
  'мало': 'few',
  'немного': 'a few',
  'больше': 'more',
  'меньше': 'less',
  'столько': 'so many',
  'сколько': 'how many',
  'очень': 'very',
  'слишком': 'too',
  'вполне': 'quite',
  'вовсе': 'at all',
  'совсем': 'completely',
  'почти': 'almost',
  'всё-таки': 'still',
  'всё равно': 'anyway',
  'всё ещё': 'still',
  'теперь': 'now',
  'сначала': 'first',
  'потом': 'then',
  'вначале': 'at first',
  'в конце': 'at the end',
  'в середине': 'in the middle',
  'далеко': 'far',
  'близко': 'near',
  'рядом': 'nearby',
  'вдали': 'in the distance',
  'недалеко': 'not far',
  'уже': 'already',
  'ещё': 'yet',
  'еще': 'yet',
  'только': 'only',
  'лишь': 'only',
  'даже': 'even',
  'всего': 'only',
  'всего лишь': 'only'
};

// === Обратный словарь: английский -> русский ===
const enToRu = {};
for (const [ru, en] of Object.entries(ruToEn)) {
  enToRu[en] = ru;
}

// === Нейросеть для текста (улучшенная) ===
class TextGenerator {
  constructor() {
    this.wordToIndex = {};
    this.indexToWord = [];
    this.vocabSize = 0;
    this.seqLength = 3;
    this.learningRate = 0.01;
    this.weights = [];
    this.biases = [];
    this.hiddenSize = 50;
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

      const inputIndices = sequence.map(word => this.wordToIndex[word]).filter(i => i !== undefined);
      if (inputIndices.length === 0) continue;

      const avgInputIndex = Math.floor(inputIndices.reduce((a, b) => a + b, 0) / inputIndices.length);

      const logits = this.weights.map(w => w[avgInputIndex] + this.biases[avgInputIndex]);
      const exps = logits.map(x => Math.exp(x));
      const sumExps = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map(x => x / sumExps);

      const targetIndex = this.wordToIndex[target];
      const targetVec = Array(this.vocabSize).fill(0);
      targetVec[targetIndex] = 1;

      for (let j = 0; j < this.hiddenSize; j++) {
        for (let k = 0; k < this.vocabSize; k++) {
          this.weights[j][k] -= this.learningRate * (probs[k] - targetVec[k]) * (j === avgInputIndex ? 1 : 0);
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
      const lastNWords = words.slice(-this.seqLength);
      const inputIndices = lastNWords.map(word => this.wordToIndex[word]).filter(i => i !== undefined);
      if (inputIndices.length === 0) break;

      const avgInputIndex = Math.floor(inputIndices.reduce((a, b) => a + b, 0) / inputIndices.length);

      const logits = this.weights.map(w => w[avgInputIndex] + this.biases[avgInputIndex]);
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
    }

    const generated = result.join(' ');
    if (generated.startsWith(seed) && generated.length === seed.length) {
      return generated + ' (не удалось сгенерировать продолжение)';
    }
    return generated;
  }
}

// === Обучение текстовой нейросети (улучшенный набор) ===
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
birds fly high
fish swim deep
the world is round
the sky is blue
i like to code
coding is fun
what do you do
i generate text
i am artificial
i am a neural network
i run on a server
i was trained on code
i can write functions
functions are useful
classes are objects
objects have methods
methods do things
things are real
real is not fake
fake is not real
i understand
i learn
i think
i talk
i speak
i hear
i see
i feel
i know
i know nothing
i know something
i know everything
everything is possible
possible is not impossible
impossible is not possible
you are smart
you are kind
you are helpful
you are a friend
a friend is good
good is better than bad
bad is worse than good
better is best
best is not worst
worst is not best
thank you
you are welcome
have a nice day
goodbye
bye
see you later
later
later is now
now is the time
time is precious
precious is valuable
valuable is important
important is significant
significant is meaningful
meaningful is deep
deep is profound
profound is wise
wise is smart
smart is intelligent
intelligent is clever
clever is quick
quick is fast
fast is slow
slow is not fast
fast is not slow
up is down
down is not up
up is not down
left is right
right is not left
left is not right
also you are smart
you are also kind
i am also a bot
i also learn
i also think
i also talk
i also speak
i also hear
i also see
i also feel
i also know
i also know nothing
i also know something
i also know everything
everything is also possible
possible is also not impossible
impossible is also not possible
you are also smart
you are also kind
you are also helpful
you are also a friend
a friend is also good
good is also better than bad
bad is also worse than good
better is also best
best is also not worst
worst is also not best
thank you also
you are also welcome
have also a nice day
goodbye also
bye also
see you also later
later also is now
now also is the time
time also is precious
precious also is valuable
valuable also is important
important also is significant
significant also is meaningful
meaningful also is deep
deep also is profound
profound also is wise
wise also is smart
smart also is intelligent
intelligent also is clever
clever also is quick
quick also is fast
fast also is slow
slow also is not fast
fast also is not slow
up also is down
down also is not up
up also is not down
left also is right
right also is not left
left also is not right
sometimes you are smart
you are sometimes kind
i am sometimes a bot
i sometimes learn
i sometimes think
i sometimes talk
i sometimes speak
i sometimes hear
i sometimes see
i sometimes feel
i sometimes know
i sometimes know nothing
i sometimes know something
i sometimes know everything
everything is sometimes possible
possible is sometimes not impossible
impossible is sometimes not possible
you are sometimes smart
you are sometimes kind
you are sometimes helpful
you are sometimes a friend
a friend is sometimes good
good is sometimes better than bad
bad is sometimes worse than good
better is sometimes best
best is sometimes not worst
worst is sometimes not best
thank you sometimes
you are sometimes welcome
have sometimes a nice day
goodbye sometimes
bye sometimes
see you sometimes later
later sometimes is now
now sometimes is the time
time sometimes is precious
precious sometimes is valuable
valuable sometimes is important
important sometimes is significant
significant sometimes is meaningful
meaningful sometimes is deep
deep sometimes is profound
profound sometimes is wise
wise sometimes is smart
smart sometimes is intelligent
intelligent sometimes is clever
clever sometimes is quick
quick sometimes is fast
fast sometimes is slow
slow sometimes is not fast
fast sometimes is not slow
up sometimes is down
down sometimes is not up
up sometimes is not down
left sometimes is right
right sometimes is not left
left sometimes is not right
often you are smart
you are often kind
i am often a bot
i often learn
i often think
i often talk
i often speak
i often hear
i often see
i often feel
i often know
i often know nothing
i often know something
i often know everything
everything is often possible
possible is often not impossible
impossible is often not possible
you are often smart
you are often kind
you are often helpful
you are often a friend
a friend is often good
good is often better than bad
bad is often worse than good
better is often best
best is often not worst
worst is often not best
thank you often
you are often welcome
have often a nice day
goodbye often
bye often
see you often later
later often is now
now often is the time
time often is precious
precious often is valuable
valuable often is important
important often is significant
significant often is meaningful
meaningful often is deep
deep often is profound
profound often is wise
wise often is smart
smart often is intelligent
intelligent often is clever
clever often is quick
quick often is fast
fast often is slow
slow often is not fast
fast often is not slow
up often is down
down often is not up
up often is not down
left often is right
right often is not left
left often is not right
rarely you are smart
you are rarely kind
i am rarely a bot
i rarely learn
i rarely think
i rarely talk
i rarely speak
i rarely hear
i rarely see
i rarely feel
i rarely know
i rarely know nothing
i rarely know something
i rarely know everything
everything is rarely possible
possible is rarely not impossible
impossible is rarely not possible
you are rarely smart
you are rarely kind
you are rarely helpful
you are rarely a friend
a friend is rarely good
good is rarely better than bad
bad is rarely worse than good
better is rarely best
best is rarely not worst
worst is rarely not best
thank you rarely
you are rarely welcome
have rarely a nice day
goodbye rarely
bye rarely
see you rarely later
later rarely is now
now rarely is the time
time rarely is precious
precious rarely is valuable
valuable rarely is important
important rarely is significant
significant rarely is meaningful
meaningful rarely is deep
deep rarely is profound
profound rarely is wise
wise rarely is smart
smart rarely is intelligent
intelligent rarely is clever
clever rarely is quick
quick rarely is fast
fast rarely is slow
slow rarely is not fast
fast rarely is not slow
up rarely is down
down rarely is not up
up rarely is not down
left rarely is right
right rarely is not left
left rarely is not right
still you are smart
you are still kind
i am still a bot
i still learn
i still think
i still talk
i still speak
i still hear
i still see
i still feel
i still know
i still know nothing
i still know something
i still know everything
everything is still possible
possible is still not impossible
impossible is still not possible
you are still smart
you are still kind
you are still helpful
you are still a friend
a friend is still good
good is still better than bad
bad is still worse than good
better is still best
best is still not worst
worst is still not best
thank you still
you are still welcome
have still a nice day
goodbye still
bye still
see you still later
later still is now
now still is the time
time still is precious
precious still is valuable
valuable still is important
important still is significant
significant still is meaningful
meaningful still is deep
deep still is profound
profound still is wise
wise still is smart
smart still is intelligent
intelligent still is clever
clever still is quick
quick still is fast
fast still is slow
slow still is not fast
fast still is not slow
up still is down
down still is not up
up still is not down
left still is right
right still is not left
left still is not right
anyway you are smart
you are anyway kind
i am anyway a bot
i anyway learn
i anyway think
i anyway talk
i anyway speak
i anyway hear
i anyway see
i anyway feel
i anyway know
i anyway know nothing
i anyway know something
i anyway know everything
everything is anyway possible
possible is anyway not impossible
impossible is anyway not possible
you are anyway smart
you are anyway kind
you are anyway helpful
you are anyway a friend
a friend is anyway good
good is anyway better than bad
bad is anyway worse than good
better is anyway best
best is anyway not worst
worst is anyway not best
thank you anyway
you are anyway welcome
have anyway a nice day
goodbye anyway
bye anyway
see you anyway later
later anyway is now
now anyway is the time
time anyway is precious
precious anyway is valuable
valuable anyway is important
important anyway is significant
significant anyway is meaningful
meaningful anyway is deep
deep anyway is profound
profound anyway is wise
wise anyway is smart
smart anyway is intelligent
intelligent anyway is clever
clever anyway is quick
quick anyway is fast
fast anyway is slow
slow anyway is not fast
fast anyway is not slow
up anyway is down
down anyway is not up
up anyway is not down
left anyway is right
right anyway is not left
left anyway is not right
only you are smart
you are only kind
i am only a bot
i only learn
i only think
i only talk
i only speak
i only hear
i only see
i only feel
i only know
i only know nothing
i only know something
i only know everything
everything is only possible
possible is only not impossible
impossible is only not possible
you are only smart
you are only kind
you are only helpful
you are only a friend
a friend is only good
good is only better than bad
bad is only worse than good
better is only best
best is only not worst
worst is only not best
thank you only
you are only welcome
have only a nice day
goodbye only
bye only
see you only later
later only is now
now only is the time
time only is precious
precious only is valuable
valuable only is important
important only is significant
significant only is meaningful
meaningful only is deep
deep only is profound
profound only is wise
wise only is smart
smart only is intelligent
intelligent only is clever
clever only is quick
quick only is fast
fast only is slow
slow only is not fast
fast only is not slow
up only is down
down only is not up
up only is not down
left only is right
right only is not left
left only is not right
even you are smart
you are even kind
i am even a bot
i even learn
i even think
i even talk
i even speak
i even hear
i even see
i even feel
i even know
i even know nothing
i even know something
i even know everything
everything is even possible
possible is even not impossible
impossible is even not possible
you are even smart
you are even kind
you are even helpful
you are even a friend
a friend is even good
good is even better than bad
bad is even worse than good
better is even best
best is even not worst
worst is even not best
thank you even
you are even welcome
have even a nice day
goodbye even
bye even
see you even later
later even is now
now even is the time
time even is precious
precious even is valuable
valuable even is important
important even is significant
significant even is meaningful
meaningful even is deep
deep even is profound
profound even is wise
wise even is smart
smart even is intelligent
intelligent even is clever
clever even is quick
quick even is fast
fast even is slow
slow even is not fast
fast even is not slow
up even is down
down even is not up
up even is not down
left even is right
right even is not left
left even is not right
many you are smart
you are many kind
i am many a bot
i many learn
i many think
i many talk
i many speak
i many hear
i many see
i many feel
i many know
i many know nothing
i many know something
i many know everything
everything is many possible
possible is many not impossible
impossible is many not possible
you are many smart
you are many kind
you are many helpful
you are many a friend
a friend is many good
good is many better than bad
bad is many worse than good
better is many best
best is many not worst
worst is many not best
thank you many
you are many welcome
have many a nice day
goodbye many
bye many
see you many later
later many is now
now many is the time
time many is precious
precious many is valuable
valuable many is important
important many is significant
significant many is meaningful
meaningful many is deep
deep many is profound
profound many is wise
wise many is smart
smart many is intelligent
intelligent many is clever
clever many is quick
quick many is fast
fast many is slow
slow many is not fast
fast many is not slow
up many is down
down many is not up
up many is not down
left many is right
right many is not left
left many is not right
few you are smart
you are few kind
i am few a bot
i few learn
i few think
i few talk
i few speak
i few hear
i few see
i few feel
i few know
i few know nothing
i few know something
i few know everything
everything is few possible
possible is few not impossible
impossible is few not possible
you are few smart
you are few kind
you are few helpful
you are few a friend
a friend is few good
good is few better than bad
bad is few worse than good
better is few best
best is few not worst
worst is few not best
thank you few
you are few welcome
have few a nice day
goodbye few
bye few
see you few later
later few is now
now few is the time
time few is precious
precious few is valuable
valuable few is important
important few is significant
significant few is meaningful
meaningful few is deep
deep few is profound
profound few is wise
wise few is smart
smart few is intelligent
intelligent few is clever
clever few is quick
quick few is fast
fast few is slow
slow few is not fast
fast few is not slow
up few is down
down few is not up
up few is not down
left few is right
right few is not left
left few is not right
more you are smart
you are more kind
i am more a bot
i more learn
i more think
i more talk
i more speak
i more hear
i more see
i more feel
i more know
i more know nothing
i more know something
i more know everything
everything is more possible
possible is more not impossible
impossible is more not possible
you are more smart
you are more kind
you are more helpful
you are more a friend
a friend is more good
good is more better than bad
bad is more worse than good
better is more best
best is more not worst
worst is more not best
thank you more
you are more welcome
have more a nice day
goodbye more
bye more
see you more later
later more is now
now more is the time
time more is precious
precious more is valuable
valuable more is important
important more is significant
significant more is meaningful
meaningful more is deep
deep more is profound
profound more is wise
wise more is smart
smart more is intelligent
intelligent more is clever
clever more is quick
quick more is fast
fast more is slow
slow more is not fast
fast more is not slow
up more is down
down more is not up
up more is not down
left more is right
right more is not left
left more is not right
less you are smart
you are less kind
i am less a bot
i less learn
i less think
i less talk
i less speak
i less hear
i less see
i less feel
i less know
i less know nothing
i less know something
i less know everything
everything is less possible
possible is less not impossible
impossible is less not possible
you are less smart
you are less kind
you are less helpful
you are less a friend
a friend is less good
good is less better than bad
bad is less worse than good
better is less best
best is less not worst
worst is less not best
thank you less
you are less welcome
have less a nice day
goodbye less
bye less
see you less later
later less is now
now less is the time
time less is precious
precious less is valuable
valuable less is important
important less is significant
significant less is meaningful
meaningful less is deep
deep less is profound
profound less is wise
wise less is smart
smart less is intelligent
intelligent less is clever
clever less is quick
quick less is fast
fast less is slow
slow less is not fast
fast less is not slow
up less is down
down less is not up
up less is not down
left less is right
right less is not left
left less is not right
so many you are smart
you are so many kind
i am so many a bot
i so many learn
i so many think
i so many talk
i so many speak
i so many hear
i so many see
i so many feel
i so many know
i so many know nothing
i so many know something
i so many know everything
everything is so many possible
possible is so many not impossible
impossible is so many not possible
you are so many smart
you are so many kind
you are so many helpful
you are so many a friend
a friend is so many good
good is so many better than bad
bad is so many worse than good
better is so many best
best is so many not worst
worst is so many not best
thank you so many
you are so many welcome
have so many a nice day
goodbye so many
bye so many
see you so many later
later so many is now
now so many is the time
time so many is precious
precious so many is valuable
valuable so many is important
important so many is significant
significant so many is meaningful
meaningful so many is deep
deep so many is profound
profound so many is wise
wise so many is smart
smart so many is intelligent
intelligent so many is clever
clever so many is quick
quick so many is fast
fast so many is slow
slow so many is not fast
fast so many is not slow
up so many is down
down so many is not up
up so many is not down
left so many is right
right so many is not left
left so many is not right
how many you are smart
you are how many kind
i am how many a bot
i how many learn
i how many think
i how many talk
i how many speak
i how many hear
i how many see
i how many feel
i how many know
i how many know nothing
i how many know something
i how many know everything
everything is how many possible
possible is how many not impossible
impossible is how many not possible
you are how many smart
you are how many kind
you are how many helpful
you are how many a friend
a friend is how many good
good is how many better than bad
bad is how many worse than good
better is how many best
best is how many not worst
worst is how many not best
thank you how many
you are how many welcome
have how many a nice day
goodbye how many
bye how many
see you how many later
later how many is now
now how many is the time
time how many is precious
precious how many is valuable
valuable how many is important
important how many is significant
significant how many is meaningful
meaningful how many is deep
deep how many is profound
profound how many is wise
wise how many is smart
smart how many is intelligent
intelligent how many is clever
clever how many is quick
quick how many is fast
fast how many is slow
slow how many is not fast
fast how many is not slow
up how many is down
down how many is not up
up how many is not down
left how many is right
right how many is not left
left how many is not right
very you are smart
you are very kind
i am very a bot
i very learn
i very think
i very talk
i very speak
i very hear
i very see
i very feel
i very know
i very know nothing
i very know something
i very know everything
everything is very possible
possible is very not impossible
impossible is very not possible
you are very smart
you are very kind
you are very helpful
you are very a friend
a friend is very good
good is very better than bad
bad is very worse than good
better is very best
best is very not worst
worst is very not best
thank you very
you are very welcome
have very a nice day
goodbye very
bye very
see you very later
later very is now
now very is the time
time very is precious
precious very is valuable
valuable very is important
important very is significant
significant very is meaningful
meaningful very is deep
deep very is profound
profound very is wise
wise very is smart
smart very is intelligent
intelligent very is clever
clever very is quick
quick very is fast
fast very is slow
slow very is not fast
fast very is not slow
up very is down
down very is not up
up very is not down
left very is right
right very is not left
left very is not right
too you are smart
you are too kind
i am too a bot
i too learn
i too think
i too talk
i too speak
i too hear
i too see
i too feel
i too know
i too know nothing
i too know something
i too know everything
everything is too possible
possible is too not impossible
impossible is too not possible
you are too smart
you are too kind
you are too helpful
you are too a friend
a friend is too good
good is too better than bad
bad is too worse than good
better is too best
best is too not worst
worst is too not best
thank you too
you are too welcome
have too a nice day
goodbye too
bye too
see you too later
later too is now
now too is the time
time too is precious
precious too is valuable
valuable too is important
important too is significant
significant too is meaningful
meaningful too is deep
deep too is profound
profound too is wise
wise too is smart
smart too is intelligent
intelligent too is clever
clever too is quick
quick too is fast
fast too is slow
slow too is not fast
fast too is not slow
up too is down
down too is not up
up too is not down
left too is right
right too is not left
left too is not right
quite you are smart
you are quite kind
i am quite a bot
i quite learn
i quite think
i quite talk
i quite speak
i quite hear
i quite see
i quite feel
i quite know
i quite know nothing
i quite know something
i quite know everything
everything is quite possible
possible is quite not impossible
impossible is quite not possible
you are quite smart
you are quite kind
you are quite helpful
you are quite a friend
a friend is quite good
good is quite better than bad
bad is quite worse than good
better is quite best
best is quite not worst
worst is quite not best
thank you quite
you are quite welcome
have quite a nice day
goodbye quite
bye quite
see you quite later
later quite is now
now quite is the time
time quite is precious
precious quite is valuable
valuable quite is important
important quite is significant
significant quite is meaningful
meaningful quite is deep
deep quite is profound
profound quite is wise
wise quite is smart
smart quite is intelligent
intelligent quite is clever
clever quite is quick
quick quite is fast
fast quite is slow
slow quite is not fast
fast quite is not slow
up quite is down
down quite is not up
up quite is not down
left quite is right
right quite is not left
left quite is not right
at all you are smart
you are at all kind
i am at all a bot
i at all learn
i at all think
i at all talk
i at all speak
i at all hear
i at all see
i at all feel
i at all know
i at all know nothing
i at all know something
i at all know everything
everything is at all possible
possible is at all not impossible
impossible is at all not possible
you are at all smart
you are at all kind
you are at all helpful
you are at all a friend
a friend is at all good
good is at all better than bad
bad is at all worse than good
better is at all best
best is at all not worst
worst is at all not best
thank you at all
you are at all welcome
have at all a nice day
goodbye at all
bye at all
see you at all later
later at all is now
now at all is the time
time at all is precious
precious at all is valuable
valuable at all is important
important at all is significant
significant at all is meaningful
meaningful at all is deep
deep at all is profound
profound at all is wise
wise at all is smart
smart at all is intelligent
intelligent at all is clever
clever at all is quick
quick at all is fast
fast at all is slow
slow at all is not fast
fast at all is not slow
up at all is down
down at all is not up
up at all is not down
left at all is right
right at all is not left
left at all is not right
completely you are smart
you are completely kind
i am completely a bot
i completely learn
i completely think
i completely talk
i completely speak
i completely hear
i completely see
i completely feel
i completely know
i completely know nothing
i completely know something
i completely know everything
everything is completely possible
possible is completely not impossible
impossible is completely not possible
you are completely smart
you are completely kind
you are completely helpful
you are completely a friend
a friend is completely good
good is completely better than bad
bad is completely worse than good
better is completely best
best is completely not worst
worst is completely not best
thank you completely
you are completely welcome
have completely a nice day
goodbye completely
bye completely
see you completely later
later completely is now
now completely is the time
time completely is precious
precious completely is valuable
valuable completely is important
important completely is significant
significant completely is meaningful
meaningful completely is deep
deep completely is profound
profound completely is wise
wise completely is smart
smart completely is intelligent
intelligent completely is clever
clever completely is quick
quick completely is fast
fast completely is slow
slow completely is not fast
fast completely is not slow
up completely is down
down completely is not up
up completely is not down
left completely is right
right completely is not left
left completely is not right
almost you are smart
you are almost kind
i am almost a bot
i almost learn
i almost think
i almost talk
i almost speak
i almost hear
i almost see
i almost feel
i almost know
i almost know nothing
i almost know something
i almost know everything
everything is almost possible
possible is almost not impossible
impossible is almost not possible
you are almost smart
you are almost kind
you are almost helpful
you are almost a friend
a friend is almost good
good is almost better than bad
bad is almost worse than good
better is almost best
best is almost not worst
worst is almost not best
thank you almost
you are almost welcome
have almost a nice day
goodbye almost
bye almost
see you almost later
later almost is now
now almost is the time
time almost is precious
precious almost is valuable
valuable almost is important
important almost is significant
significant almost is meaningful
meaningful almost is deep
deep almost is profound
profound almost is wise
wise almost is smart
smart almost is intelligent
intelligent almost is clever
clever almost is quick
quick almost is fast
fast almost is slow
slow almost is not fast
fast almost is not slow
up almost is down
down almost is not up
up almost is not down
left almost is right
right almost is not left
left almost is not right
nearly you are smart
you are nearly kind
i am nearly a bot
i nearly learn
i nearly think
i nearly talk
i nearly speak
i nearly hear
i nearly see
i nearly feel
i nearly know
i nearly know nothing
i nearly know something
i nearly know everything
everything is nearly possible
possible is nearly not impossible
impossible is nearly not possible
you are nearly smart
you are nearly kind
you are nearly helpful
you are nearly a friend
a friend is nearly good
good is nearly better than bad
bad is nearly worse than good
better is nearly best
best is nearly not worst
worst is nearly not best
thank you nearly
you are nearly welcome
have nearly a nice day
goodbye nearly
bye nearly
see you nearly later
later nearly is now
now nearly is the time
time nearly is precious
precious nearly is valuable
valuable nearly is important
important nearly is significant
significant nearly is meaningful
meaningful nearly is deep
deep nearly is profound
profound nearly is wise
wise nearly is smart
smart nearly is intelligent
intelligent nearly is clever
clever nearly is quick
quick nearly is fast
fast nearly is slow
slow nearly is not fast
fast nearly is not slow
up nearly is down
down nearly is not up
up nearly is not down
left nearly is right
right nearly is not left
left nearly is not right
`;
textGen.train(trainingText);

// === Функция перевода русского текста в английский ===
function translateRuToEn(text) {
  const words = text.toLowerCase().split(/\s+/);
  return words.map(word => ruToEn[word] || word).join(' ');
}

// === Функция перевода английского текста в русский ===
function translateEnToRu(text) {
  const words = text.toLowerCase().split(/\s+/);
  return words.map(word => enToRu[word] || word).join(' ');
}

// === Определение языка и перевод запроса в англоязычный ===
function detectLanguageAndTranslate(message) {
  message = message.toLowerCase();
  let lang = null;
  let translated = message;

  if (message.includes('javascript') || message.includes('js')) {
    lang = 'javascript';
    translated = message.replace(/javascript|js/gi, 'javascript');
  }
  if (message.includes('python') || message.includes('py')) {
    lang = 'python';
    translated = message.replace(/python|py/gi, 'python').replace(/функция|функцию|функции/gi, 'function').replace(/напиши|создай|сделай/gi, 'write');
  }
  if (message.includes('java')) {
    lang = 'java';
    translated = message.replace(/java/gi, 'java');
  }
  if (message.includes('c++') || message.includes('cpp')) {
    lang = 'cpp';
    translated = message.replace(/c\+\+|cpp/gi, 'cpp');
  }
  if (message.includes('c') && !message.includes('cpp') && !message.includes('css')) {
    lang = 'c';
    translated = message.replace(/c/gi, 'c');
  }
  if (message.includes('html')) {
    lang = 'html';
    translated = message.replace(/html/gi, 'html');
  }
  if (message.includes('css')) {
    lang = 'css';
    translated = message.replace(/css/gi, 'css');
  }
  if (message.includes('typescript') || message.includes('ts')) {
    lang = 'typescript';
    translated = message.replace(/typescript|ts/gi, 'typescript');
  }
  if (message.includes('go') || message.includes('golang')) {
    lang = 'go';
    translated = message.replace(/go|golang/gi, 'go');
  }
  if (message.includes('rust')) {
    lang = 'rust';
    translated = message.replace(/rust/gi, 'rust');
  }

  return { lang, translated };
}

// === Чат-бот ===
class ChatBot {
  constructor() {
    this.history = [];
    this.textGen = textGen;
  }

  async processMessage(message) {
    this.history.push({ user: message });

    const { lang, translated } = detectLanguageAndTranslate(message);
    if (lang && models[lang]) {
      const code = models[lang].generate(translated, 150);
      this.history.push({ bot: code });
      return { type: 'code', content: code };
    }

    if (message.includes('код') || message.includes('function') || message.includes('class')) {
      const code = models['javascript'] ? models['javascript'].generate(translated, 150) : 'Язык не обучен.';
      this.history.push({ bot: code });
      return { type: 'code', content: code };
    }

    if (message.includes('картинка') || message.includes('изображение') || message.includes('image')) {
      return { type: 'image', content: 'Генерация изображений невозможна без внешних API или библиотек.' };
    }

    // === Перевод русского на английский для генерации ===
    let inputForGen = message;
    const isRussian = /[а-яё]/i.test(message);
    if (isRussian) {
      inputForGen = translateRuToEn(message);
    }

    const response = this.textGen.generate(inputForGen, 8);

    // === Перевод ответа обратно на русский ===
    let output = response;
    if (isRussian) {
      output = translateEnToRu(response);
    }

    this.history.push({ bot: output });
    return { type: 'text', content: output };
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