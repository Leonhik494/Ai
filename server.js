const express = require('express');
const path = require('path');
const { pipeline } = require('@xenova/transformers');

const app = express();
const PORT = process.env.PORT || 10000;

// --- Загрузка модели ---
// Используем глобальную переменную для хранения конвейера
let generator = null;

async function loadModel() {
    console.log('Загрузка модели...');
    // Загружаем конвейер генерации текста
    // Используем gpt2-medium, который меньше, чем gpt2, но все еще требует ресурсов
    // Попробуйте gpt2, если gpt2-medium не поместится
    generator = await pipeline('text-generation', 'Xenova/gpt2');
    // Для еще меньшей модели, если gpt2 все равно слишком большая:
    // generator = await pipeline('text-generation', 'Xenova/distilgpt2');
    console.log('Модель загружена.');
}

// --- Веб-сервер ---
// Middleware для парсинга JSON
app.use(express.json());

// Middleware для CORS
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Content-Type');
    if (req.method === 'OPTIONS') {
        res.sendStatus(204);
    } else {
        next();
    }
});

// Маршрут для получения frontend.html
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'frontend.html'));
});

// Маршрут для генерации текста
app.post('/generate', async (req, res) => {
    if (!generator) {
        return res.status(503).json({ error: 'Модель еще не загружена.' });
    }

    try {
        const { input } = req.body;
        if (typeof input !== 'string' || input.trim() === '') {
            return res.status(400).json({ error: 'Invalid input: expected a non-empty string.' });
        }

        console.log(`Генерация для: "${input}"`);
        // Генерируем текст
        // max_new_tokens - максимальное количество новых токенов (слов/субслов) для генерации
        // temperature - влияет на "креативность", 1.0 - стандарт, 0.5 - более предсказуемо
        // do_sample - использовать ли сэмплирование (true) или greedy (false)
        const output = await generator(input.trim(), {
            max_new_tokens: 50,
            temperature: 0.7,
            do_sample: true,
            // truncation: true, // Не всегда нужно для генерации
            // padding: true,   // Не всегда нужно для генерации
        });

        console.log('Сгенерированный текст:', output);
        // Вывод обычно содержит массив объектов, берем первый результат
        const generatedText = output[0].generated_text;

        // Возвращаем только сгенерированную часть (без исходного ввода)
        // Поскольку модель генерирует вместе с инпутом, нужно обрезать
        const responseText = generatedText.substring(input.length).trim();

        res.json({ output: responseText });
    } catch (error) {
        console.error("Error processing generate request:", error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

// Запускаем сервер после загрузки модели
async function startServer() {
    await loadModel(); // Ждем загрузки модели
    app.listen(PORT, () => {
        console.log(`Server is running on port ${PORT}`);
    });
}

startServer().catch(err => {
    console.error("Failed to start server:", err);
    process.exit(1);
});