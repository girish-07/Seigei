const openai = require('openai');
const express = require('express');
const app = express();
const dotenv = require('dotenv');
dotenv.config();

const openaiConfig = new openai.OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});

const port = process.env.PORT;

app.get('/', (req, res) => {
    res.send("HELLO WORLD");
})

app.get('/gpt', async (req, res) => {
    const words = "my school friend come home food drink";
    const messages = "Make a meaningful sentence out of the phrases: " + words;
    console.log(messages);
    const completion = await openaiConfig.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: [
            {"role": "user", "content": messages}
        ],
        max_tokens: 200
    })
    console.log(completion);
    res.send(completion.choices[0].message);

})

app.listen(port, () => {
    console.log("Server started");
    console.log(port);
})