import * as dotenv from "dotenv";
dotenv.config();

import { createInterface } from "readline/promises";
import { stdin as input, stdout as output } from "node:process";

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { TavilySearch } from "@langchain/tavily";
import { createRetrieverTool } from "langchain/tools/retriever";
import { createToolCallingAgent, AgentExecutor } from "langchain/agents";
import { BufferMemory } from "langchain/memory";
import { UpstashRedisChatMessageHistory } from "@langchain/community/stores/message/upstash_redis";

// === 1. Initialize LLM ===
const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GOOGLE_API_KEY,
  model: "gemini-1.5-flash",
  temperature: 0.7,
});

// === 2. Long-term memory ===
const upstashHistory = new UpstashRedisChatMessageHistory({
  sessionId: "mysession",
  config: {
    url: process.env.UPSTASH_REDIS_REST_URL,
    token: process.env.UPSTASH_REDIS_REST_TOKEN,
  },
});

// === 2. Initialize memory ===
const memory = new BufferMemory({
  memoryKey: "chat_history",
  chatHistory: upstashHistory,
});

// === 3. Load & index website docs ===
const loader = new CheerioWebBaseLoader(
  process.env.WEB_URL || "https://python.langchain.com/docs/concepts/lcel/"
);
const webDocs = await loader.load();

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 100,
});

const webChunks = await splitter.splitDocuments(webDocs);

const webEmb = new HuggingFaceInferenceEmbeddings({
  modelName: "sentence-transformers/all-MiniLM-L6-v2",
  apiKey: process.env.HF_API_KEY,
});
const webVec = await MemoryVectorStore.fromDocuments(webChunks, webEmb);
const retriever = webVec.asRetriever({ k: 2 });

// === 4. RAG Chain ===
const retrieverPrompt = ChatPromptTemplate.fromMessages([
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
  ["user", "Generate a search query to find relevant info in the docs:"],
]);
const retrieverChain = await createHistoryAwareRetriever({
  llm: model,
  retriever,
  rephrasePrompt: retrieverPrompt,
});
const ragPrompt = ChatPromptTemplate.fromMessages([
  ["system", "Use the following context to answer: {context}"],
  new MessagesPlaceholder("chat_history"),
  ["user", "{input}"],
]);
const ragChain = await createStuffDocumentsChain({
  llm: model,
  prompt: ragPrompt,
});
const ragQAChain = await createRetrievalChain({
  combineDocsChain: ragChain,
  retriever: retrieverChain,
});
// === 5. Tools (Web + Docs Search) ===
const tools = [];
let searchTool;

// Use a try-catch block for robust initialization of the TavilySearch tool
try {
  searchTool = new TavilySearch({
    apikey: process.env.TAVILY_API_KEY,
    name: "search_tool",
    description: "Use this tool to search the web for real-time or up-to-date information, such as news, weather, current events, or anything not found in the provided documents.",
  });
} catch (error) {
  console.error("Failed to initialize TavilySearch tool:", error.message);
  // Do not re-throw, just continue without the tool
}

const retrieverTool = createRetrieverTool({
  retriever,
  name: "website_search",
  description: "Use this tool to answer questions related to website content sent by me",
});

// Add tools to the array, only if they were initialized successfully
tools.push(retrieverTool);
if (searchTool && typeof searchTool.name === 'string') {
  tools.push(searchTool);
}


// === 6. Agent (Gemini-compatible) ===
const agentPrompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant called Max. You can answer using the tools provided, including real-time search."],
  ["system", "If you don't know the answer, you can search the web or use the docs."],
  new MessagesPlaceholder("chat_history"),
  ["human", "{input}"],
  new MessagesPlaceholder("agent_scratchpad"),
]);

const agent = await createToolCallingAgent({
  llm: model,
  prompt: agentPrompt,
  tools,
});

const agentExecutor = new AgentExecutor({
  agent,
  tools,
  verbose: true,
});

// === 7. CLI Chat Loop ===
const rl = createInterface({ input, output });
console.log("🧠 Max is online. Type 'exit' to quit, or 'search your question' to use the agent.");

const chatHistory = [];

async function askLoop() {
  const userInput = await rl.question("> ");
  if (userInput.toLowerCase() === "exit") {
    console.log("👋 Goodbye!");
    rl.close();
    return;
  }

  let response;
  if (userInput.startsWith("search")) {
    const query = userInput.slice(6).trim();
    console.log("🔍 Searching for:", query);
    response = await agentExecutor.invoke({
      input: query,
      chat_history: chatHistory,
    });
  } else {
    response = await ragQAChain.invoke({
      input: userInput,
      chat_history: chatHistory,
    });
  }

  const outputText = response.output ?? response.answer ?? response;
  console.log("🗣 Max:", outputText);

  chatHistory.push(new HumanMessage(userInput));
  chatHistory.push(new AIMessage(outputText));
  await memory.saveContext({ input: userInput }, { output: outputText });

  askLoop();
}

askLoop();
