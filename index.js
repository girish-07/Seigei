const openai = require('openai');
const express = require('express');
const app = express();
const dotenv = require('dotenv');
const admin = require('firebase-admin');
const serviceAccount = require("./serviceAccountKey.json")
dotenv.config();

admin.initializeApp({
    credential: admin.credential.cert(serviceAccount)
})

let db = admin.firestore();

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

app.get('/db', async(req, res) => {
    var wordSet = "i sleep now"
    var description = "hello all!"
    product = await db.collection("WordSummary").get();
    var items = new Map([])
    product.forEach((doc) => { items.set(doc.data().WordSet, doc.data().Description) })
    if(!items.has(wordSet)) {
        const words = wordSet;
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
        description = completion.choices[0].message.content
        await db.collection("WordSummary").doc().set({WordSet: wordSet, Description: description});
        items.set(wordSet, description);
    }
    else {
        description = items.get(wordSet);
    }
    items.set(wordSet, description);
    console.log(items);
    res.send(description);
})

app.listen(port, () => {
    console.log("Server started");
    console.log(port);
})