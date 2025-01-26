---
title: "Build AI Text Classification in Rust with Rig"
source: "https://dev.to/0thtachi/build-ai-text-classification-in-rust-with-rig-ajl"
author:
  - "[[Tachi 0x]]"
published: 2024-09-17
created: 2024-12-23
description: "TL;DR: This guide walks you through building a text classification system in Rust using the Rig... Tagged with rust, ai, llm, opensource."
tags:
  - "clippings"
---
**TL;DR:** This guide walks you through building a text classification system in Rust using the Rig library. In very few lines of code, you'll create a system that performs sentiment analysis and classifies news articles by topic, leveraging OpenAI's GPT models for accurate text classification.

## Introduction

Text classification is a fundamental task in natural language processing, involving the assignment of predefined categories to text documents. It's widely used in applications such as sentiment analysis, content categorization, and spam detection. Large Language Models (LLMs) have significantly improved the accuracy and flexibility of text classification tasks, but working with them can be complex.

Rig, an open-source Rust library, simplifies the development of LLM-powered applications, including text classification systems. In this guide, I'll walk you through the process of building a functional text classification system using Rig. We'll create a system capable of performing sentiment analysis and classifying news articles by topic, demonstrating Rig's application to real-world text classification tasks.

> 💡 **Tip**: New to Rig?
> 
> If you're not familiar with Rig or want a comprehensive introduction to its capabilities, check out our introductory blog post: [Rig: A Rust Library for Building LLM-Powered Applications](https://dev.to/0thtachi/rig-a-rust-library-for-building-llm-powered-applications-3g75). It provides an overview of Rig's features and how it simplifies LLM application development in Rust.
> 
> 💡 **Tip**: New to Rust?
> 
> This guide assumes some familiarity with Rust and a set-up coding environment. If you're just starting out or need to set up your environment, check out these quick guides:
> 
> - [Introduction to Rust](https://www.rust-lang.org/learn)
> - [Setting up Rust with VS Code](https://users.rust-lang.org/t/setting-up-rust-with-vs-code/76907)
> 
> These resources will help you get up to speed quickly!

## Setting Up the Project

Let's start by setting up our Rust project and installing the necessary dependencies.

Create a new Rust project:  

```
cargo new text_classifier
cd text_classifier
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Add the following dependencies to your `Cargo.toml`:  

```
[dependencies]
rig-core = "0.0.6"
tokio = { version = "1.34.0", features = ["full"] }
anyhow = "1.0.75"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
dotenv = "0.15.0"
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

These dependencies provide the core functionality we need:

- `rig-core`: The main Rig library for LLM applications
- `tokio`: Asynchronous runtime for Rust
- `anyhow`: Flexible error handling
- `serde` and `serde_json`: JSON serialization and deserialization
- `dotenv`: Loading environment variables from a file

Before coding, set up your OpenAI API key:  

```
export OPENAI_API_KEY=your_api_key_here
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

## Building the Text Classification System

We'll start with a simple sentiment analysis classifier. This will demonstrate the basics of using Rig for text classification.

First, let's define our data structures:  

```
// Import necessary dependencies
use rig::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// Define an enum to represent sentiment categories
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
enum Sentiment {
    Positive,
    Negative,
    Neutral,
}

// Define a struct to hold the sentiment classification result
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct SentimentClassification {
    sentiment: Sentiment,
    confidence: f32,
}

fn pretty_print_result(text: &str, result: &SentimentClassification) {
    println!("Text: \"{}\"", text);
    println!("Sentiment Analysis Result:");
    println!("  Sentiment: {:?}", result.sentiment);
    println!("  Confidence: {:.2}%", result.confidence * 100.0);
    println!();
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Now, let's implement our sentiment classifier:  

```
#[tokio::main]
async fn main() {
    // Initialize the OpenAI client
    let openai_client = openai::Client::from_env();

    // Create a sentiment classifier using Rig's Extractor
    let sentiment_classifier = openai_client
        .extractor::<SentimentClassification>("gpt-3.5-turbo")
        .preamble("
            You are a sentiment analysis AI. Classify the sentiment of the given text.
            Respond with Positive, Negative, or Neutral, along with a confidence score (0-1).
            Examples:
            Text: 'This movie was terrible. I hated every minute of it.'
            Result: Negative, 0.9
            Text: 'The weather today is okay, nothing special.'
            Result: Neutral, 0.7
            Text: 'I'm so excited about my upcoming vacation!'
            Result: Positive, 0.95
        ")
        .build();

    // Sample text to classify
    let text = "I absolutely loved the new restaurant. The food was amazing!";

    // Perform sentiment classification
    match sentiment_classifier.extract(text).await {
        Ok(result) => pretty_print_result(text, &result),
        Err(e) => eprintln!("Error classifying sentiment: {}", e),
    }
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This code creates a sentiment classifier using OpenAI's GPT-3.5-turbo model. The `Extractor` is configured with a preamble that instructs the model to perform sentiment analysis and provides examples to guide its output. When we call `extract` with our input text, the model classifies the sentiment and returns a `SentimentClassification` struct.

When you run this code, you should see output similar to this:  

```
Text: I absolutely loved the new restaurant. The food was amazing!
Sentiment Analysis Result:
  Sentiment: Positive
  Confidence: 95.00%
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

The output demonstrates that our sentiment classifier correctly identified the positive sentiment in the input text. The high confidence score (0.95) indicates that the model is very certain about its classification. This aligns well with the strongly positive language used in the input text.

## Advanced Text Classification: News Article Classifier

Now that we've covered the basics, let's build a more complex system: a news article classifier that categorizes articles by topic and performs sentiment analysis.

First, let's define our structures:  

```
// Import necessary dependencies
use rig::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// Define enum for sentiment classification
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
enum Sentiment {
    Positive,
    Negative,
    Neutral,
}

// Define an enum for sentiment confidence level
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct SentimentClassification {
    sentiment: Sentiment,
    confidence: f32,
}

// Define an enum to represent news article topics
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
enum Topic {
    Politics,
    Technology,
    Sports,
    Entertainment,
    Other(String),
}

// Define a struct to hold the news article classification result
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct NewsArticleClassification {
    topic: Topic,
    sentiment: SentimentClassification,
    summary: String,
}

fn pretty_print_result(article: &str, result: &NewsArticleClassification) {
    println!("Article: \"{}...\"", &article[..100]); // Print first 100 characters
    println!("Classification Result:");
    println!("  Topic: {:?}", result.topic);
    println!("  Sentiment: {:?}", result.sentiment.sentiment);
    println!("  Confidence: {:.2}%", result.sentiment.confidence * 100.0);
    println!("  Summary: {}", result.summary);
    println!();
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Now, let's implement our news article classifier:  

```
#[tokio::main]
async fn main() {
    // Initialize the OpenAI client
    let openai_client = openai::Client::from_env();

    // Create a news article classifier using Rig's Extractor
    let news_classifier = openai_client
        .extractor::<NewsArticleClassification>("gpt-4")
        .preamble("
            You are a news article classification AI. For the given news article:
            1. Classify the main topic (Politics, Technology, Sports, Entertainment, or Other).
            2. Analyze the overall sentiment (Positive, Negative, or Neutral) with a confidence score.
            3. Provide a brief summary of the article.
        ")
        .build();

   // Sample news article to classify
    let article = "
        After conducting the first-ever commercial spacewalk and traveling farther from Earth than anyone \
        in more than half a century, the astronauts of the Polaris Dawn mission returned to Earth safely \
        early Sunday.

        The SpaceX Crew Dragon capsule splashed down in the Gulf of Mexico, off the coast of Dry Tortugas, \
        Fla., shortly after 3:30 a.m., carrying Jared Isaacman, a billionaire entrepreneur, and his crew \
        of three private astronauts, according to a SpaceX livestream.

        The ambitious space mission, a collaboration between Mr. Isaacman and Elon Musk's SpaceX, spent \
        five days in orbit, achieved several milestones in private spaceflight and was further evidence \
        that space travel and spacewalks are no longer the exclusive domain of professional astronauts \
        working at government agencies like NASA.

        The Crew Dragon capsule launched on Tuesday, after delays because of a helium leak and bad weather. \
        On board were Mr. Isaacman, the mission commander and the founder of the payment services company \
        Shift4; Sarah Gillis and Anna Menon, SpaceX employees; and Scott Poteet, a retired U.S. Air Force \
        lieutenant colonel.

        Late on Tuesday, its orbit reached a high point of about 870 miles above the Earth's surface. That \
        beat the record distance for astronauts on a mission not headed to the moon, which the Gemini XI \
        mission set in 1966 at 853 miles high, and made Ms. Gillis and Ms. Menon the first women ever to \
        fly so far from Earth.

        On Thursday, Mr. Isaacman and Ms. Gillis became the first private astronauts to successfully complete \
        a spacewalk. The operation involved the crew letting all the air out of the spacecraft, because it \
        had no airlock, while the other two crew members wore spacesuits inside the airless capsule. Mr. \
        Isaacman moved outside and conducted mobility tests of his spacesuit for a few minutes before \
        re-entering the capsule. Ms Gillis then moved outside and performed the same tests.

        This was the first of three Polaris missions aimed at accelerating technological advances needed to \
        fulfill Mr. Musk's dream of sending people to Mars someday. A key goal of the mission was to further \
        the development of more advanced spacesuits that would be needed for SpaceX to try any future \
        off-world colonization.

        During a news conference before the launch, Mr. Isaacman mused that one day, someone might step onto \
        Mars wearing a version of the spacesuit that SpaceX had developed for this flight. Closer to Earth, \
        commercial spacewalks also present other possibilities, like technicians repairing private satellites \
        in orbit.

        During the spaceflight, the four astronauts conducted about 40 experiments, mostly about how \
        weightlessness and radiation affect the human body. They also tested laser communications between \
        the Crew Dragon and SpaceX's constellation of Starlink internet satellites.\
    ";

    // Perform news article classification
    match news_classifier.extract(article).await {
        Ok(result) => pretty_print_result(article, &result),
        Err(e) => eprintln!("Error classifying article: {}", e),
    }
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

When you run this code, you might see output like this:  

```
Article: "
        After conducting the first-ever commercial spacewalk and traveling farther from Earth than ..."
Classification Result:
  Topic: Technology
  Sentiment: Positive
  Confidence: 90.00%
  Summary: The SpaceX Crew Dragon capsule carrying billionaire entrepreneur Jared Isaacman and his crew of three private astronauts returned successfully to Earth, after conducting the first-ever commercial spacewalk and setting a new distance record. The mission, a collaboration between Isaacman and SpaceX, is a part of three Polaris missions aimed to accelerate technological advances for space colonization. SpaceX also hopes to develop more advanced spacesuits necessary for future Mars missions.
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This output shows that our news article classifier has successfully categorized the article as belonging to the Technology topic. The sentiment is classified as Positive with a relatively high confidence of 90%.

The classification makes sense given the article's content:

1. The topic is clearly technology-focused, discussing space exploration, SpaceX's Crew Dragon capsule, and advancements in spacesuit technology.
2. The positive sentiment is reflected in phrases like "first-ever commercial spacewalk," "achieved several milestones," and "accelerating technological advances." The high confidence in the positive sentiment comes from the overall optimistic tone about the mission's achievements and future possibilities.
3. The technology classification is appropriate as the article primarily focuses on the technological aspects of space exploration, including the spacecraft, spacesuits, and experiments conducted during the mission.
4. While the article does mention some challenges (like delays due to a helium leak and bad weather), these are presented as minor setbacks in an otherwise successful mission, which explains why the sentiment remains positive but the confidence isn't at the maximum.

This example demonstrates how our classifier can handle complex, real-world news articles, extracting both the main topic and the overall sentiment accurately. It shows Rig's capability to process nuanced content and provide insightful classifications.

## Best Practices and Common Pitfalls

As you work with Rig for text classification, keep these best practices in mind:

1. **Prompting**: Craft clear, specific prompts. Include examples in your preamble to guide the model's output.
2. **Model Selection**: Choose the appropriate model for your task. While GPT-4 is more capable, GPT-3.5-turbo may be sufficient for many classification tasks and is more cost-effective.
3. **Error Handling**: Always handle potential errors from API calls and unexpected model outputs.
4. **Validation**: Implement output validation to ensure the model's responses match your expected format.
5. **Batching**: Use batching for processing multiple texts to reduce API calls and improve efficiency.

Watch out for these common pitfalls:

1. **Overreliance on the Model**: Don't assume the model will always produce perfect classifications. Implement checks and balances in your system.
2. **Ignoring Rate Limits**: Be aware of and respect the rate limits of your LLM provider.
3. **Neglecting Security**: Always protect your API keys and sensitive data.
4. **Lack of Monitoring**: Implement proper logging and monitoring to catch issues early.

## Conclusion and Next Steps

In this guide, we've built a text classification system using Rig, demonstrating its capability to simplify LLM-powered applications in Rust. By creating both a sentiment analyzer and a news article classifier, we've showcased Rig's flexibility and power in handling diverse text classification tasks.

To further explore Rig's capabilities:

1. Adapt the provided code examples to your specific use cases.
2. Dive deeper into Rig's features by exploring the [documentation](https://docs.rs/rig-core/latest/rig/).
3. Experiment with different models and classification tasks.

For additional resources and community engagement:

- Browse more examples in our [gallery](https://github.com/0xPlaygrounds/awesome-rig/tree/main).
- Contribute or report issues on [GitHub](https://github.com/0xPlaygrounds/rig).
- Join discussions in our [Discord community](https://discord.com/invite/playgrounds).

We're continually improving Rig based on user feedback. If you build a project with Rig, consider sharing your experience through our [feedback form](https://bit.ly/Rig-Review) and get rewarded with $100. Your insights are valuable in shaping Rig's future development.

Happy coding!---
title: "Build a Fast and Lightweight Rust Vector Search App with Rig & LanceDB"
source: "https://dev.to/0thtachi/build-a-fast-and-lightweight-rust-vector-search-app-with-rig-lancedb-57h2"
author:
  - "[[Tachi 0x]]"
published: 2024-11-21
created: 2024-12-23
description: "TL;DR: Build a powerful semantic search system in Rust using Rig and LanceDB. We'll guide you... Tagged with rust, llm, ai, opensource."
tags:
  - "clippings"
---
**TL;DR**: Build a powerful semantic search system in Rust using Rig and LanceDB. We'll guide you step-by-step through creating, storing, and searching vector embeddings efficiently with hands-on examples. Perfect for building RAG systems, semantic search engines, and more.

## Introduction

Semantic search is transforming the way we find and understand information. Unlike traditional keyword search, it captures the intent behind your queries, offering a more nuanced retrieval process. However, building these systems can feel daunting, often involving complex embeddings, vector databases, and similarity search algorithms.

That's where LanceDB comes in.

### Why LanceDB?

[LanceDB](https://lancedb.github.io/lancedb/) is an open-source vector database tailored for AI applications and vector search. It provides:

- **Embedded Database**: Works directly in your application without needing external servers.
- **High Performance**: Leverages Arrow format for efficient data storage and retrieval.
- **Scalable**: Handles terabyte-scale datasets efficiently.
- **Vector Indexing**: Supports both exact and approximate nearest neighbor searches out of the box.

Combined with Rig's embedding and LLM capabilities, you can create a powerful, efficient semantic search solution with minimal code.

Let's dive in!

> You can find the full source code for this project in our [GitHub repo](https://github.com/0xPlaygrounds/rig/tree/main/rig-lancedb).

## Prerequisites

Before we begin, make sure you have:

- Rust installed ([rust-lang.org](https://rust-lang.org/))
- An OpenAI API key ([platform.openai.com](https://platform.openai.com/))
- Basic familiarity with Rust and asynchronous programming

## Project Setup

To start, create a new Rust project:  

```
cargo new vector_search
cd vector_search
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Update your `Cargo.toml` to add the necessary dependencies:  

```
[dependencies]
rig-core = "0.4.0"
rig-lancedb = "0.1.1"
lancedb = "0.10.0"
tokio = { version = "1.40.0", features = ["full"] }
anyhow = "1.0.89"
futures = "0.3.30"
serde = { version = "1.0.210", features = ["derive"] }
serde_json = "1.0.128"
arrow-array = "52.2.0"
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Here’s a quick overview of each dependency:

- `rig-core` and `rig-lancedb`: The core libraries for embedding generation and vector search.
- `lancedb`: The embedded vector database.
- `tokio`: Asynchronous runtime support.
- `arrow-array`: To work with Arrow's columnar format, which LanceDB uses internally.
- Others for error handling, serialization, and futures support.

Now, create a `.env` file to store your OpenAI API key:  

```
echo "OPENAI_API_KEY=your_key_here" > .env
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

## Building the Search System

We’ll break this into manageable steps. First, let’s create a utility function to handle data conversion between Rig's embeddings and LanceDB's format.

Create `src/utils.rs`:  

```
use std::sync::Arc;
use arrow_array::{
    types::Float64Type, ArrayRef, FixedSizeListArray,
    RecordBatch, StringArray
};
use lancedb::arrow::arrow_schema::{DataType, Field, Fields, Schema};
use rig::embeddings::DocumentEmbeddings;

// Define the schema for our LanceDB table
pub fn schema(dims: usize) -> Schema {
    Schema::new(Fields::from(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float64, true)),
                dims as i32,
            ),
            false,
        ),
    ]))
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This schema function defines the structure of our table:

- **`id`**: A unique identifier for each document.
- **`content`**: The text content of the document.
- **`embedding`**: The vector representation of the content.
- **`dims`** parameter: Represents the size of embedding vectors (e.g., 1536 for OpenAI's `ada-002` model).

Next, add the conversion function to convert `DocumentEmbeddings` into `RecordBatch` for LanceDB:  

```
pub fn as_record_batch(
    records: Vec<DocumentEmbeddings>,
    dims: usize,
) -> Result<RecordBatch, lancedb::arrow::arrow_schema::ArrowError> {
    let id = StringArray::from_iter_values(
        records
            .iter()
            .flat_map(|record| (0..record.embeddings.len())
                .map(|i| format!("{}-{i}", record.id)))
            .collect::<Vec<_>>(),
    );

    let content = StringArray::from_iter_values(
        records
            .iter()
            .flat_map(|record| {
                record
                    .embeddings
                    .iter()
                    .map(|embedding| embedding.document.clone())
            })
            .collect::<Vec<_>>(),
    );

    let embedding = FixedSizeListArray::from_iter_primitive::<Float64Type, _, _>(
        records
            .into_iter()
            .flat_map(|record| {
                record
                    .embeddings
                    .into_iter()
                    .map(|embedding| embedding.vec.into_iter().map(Some).collect::<Vec<_>>())
                    .map(Some)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        dims as i32,
    );

    RecordBatch::try_from_iter(vec![
        ("id", Arc::new(id) as ArrayRef),
        ("content", Arc::new(content) as ArrayRef),
        ("embedding", Arc::new(embedding) as ArrayRef),
    ])
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This function is crucial as it converts our Rust data structures into Arrow's columnar format, which LanceDB uses internally:

1. Creates string arrays for IDs and content.
2. Converts embeddings into fixed-size lists.
3. Assembles everything into a `RecordBatch`.

With our utility functions ready, let’s build the main search functionality in `src/main.rs`. We’ll implement this step-by-step, explaining each part along the way.

### Setting Up Dependencies

First, let’s import the required libraries:  

```
use anyhow::Result;
use arrow_array::RecordBatchIterator;
use lancedb::{index::vector::IvfPqIndexBuilder, DistanceType};
use rig::{
    embeddings::{DocumentEmbeddings, EmbeddingModel, EmbeddingsBuilder},
    providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
    vector_store::VectorStoreIndex,
};
use rig_lancedb::{LanceDbVectorStore, SearchParams};
use serde::Deserialize;
use std::{env, sync::Arc};

mod utils;
use utils::{as_record_batch, schema};
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

These imports bring in:

- Rig’s embedding and vector storage tools.
- LanceDB’s database capabilities.
- Arrow data structures for efficient processing.
- Utilities for serialization, error handling, and async programming.

### Defining Data Structures

We’ll create a simple struct to represent our search results:  

```
#[derive(Debug, Deserialize)]
struct SearchResult {
    content: String,
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This struct maps to database records, representing the content we want to retrieve.

### Generating Embeddings

Generating document embeddings is a core part of our system. Let’s implement this function:  

```
async fn create_embeddings(client: &Client) -> Result<Vec<DocumentEmbeddings>> {
    let model = client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Set up dummy data to meet the 256 row requirement for IVF-PQ indexing
    let dummy_doc = "Let there be light".to_string();
    let dummy_docs = vec![dummy_doc; 256];

    // Generate embeddings for the data
    let embeddings = EmbeddingsBuilder::new(model)
        // First add our real documents
        .simple_document(
            "doc1",
            "Rust provides zero-cost abstractions and memory safety without garbage collection.",
        )
        .simple_document(
            "doc2",
            "Python emphasizes code readability with significant whitespace.",
        )
        // Add dummy documents to meet minimum requirement using enumerate to generate unique IDs
        .simple_documents(
            dummy_docs
                .into_iter()
                .enumerate()
                .map(|(i, doc)| (format!("doc{}", i + 3), doc))
                .collect(),
        )
        .build()
        .await?;

    Ok(embeddings)
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This function handles:

1. Initializing the OpenAI embedding model.
2. Creating embeddings for our real documents.
3. Adding dummy data to meet LanceDB’s indexing requirements.

### Configuring the Vector Store

Now, let’s set up LanceDB and configure it with appropriate indexing and search parameters:  

```
async fn setup_vector_store<M: EmbeddingModel>(
    embeddings: Vec<DocumentEmbeddings>,
    model: M,
) -> Result<LanceDbVectorStore<M>> {
    // Initialize LanceDB
    let db = lancedb::connect("data/lancedb-store").execute().await?;

    // Drop the existing table if it exists - important for development
    if db
        .table_names()
        .execute()
        .await?
        .contains(&"documents".to_string())
    {
        db.drop_table("documents").await?;
    }

    // Create table with embeddings
    let record_batch = as_record_batch(embeddings, model.ndims())?;
    let table = db
        .create_table(
            "documents",
            RecordBatchIterator::new(vec![Ok(record_batch)], Arc::new(schema(model.ndims()))),
        )
        .execute()
        .await?;

    // Create an optimized vector index using IVF-PQ
    table
        .create_index(
            &["embedding"],
            lancedb::index::Index::IvfPq(
                IvfPqIndexBuilder::default().distance_type(DistanceType::Cosine),
            ),
        )
        .execute()
        .await?;

    // Configure search parameters
    let search_params = SearchParams::default().distance_type(DistanceType::Cosine);

    // Create and return vector store
    Ok(LanceDbVectorStore::new(table, model, "id", search_params).await?)
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This setup function:

1. Connects to the LanceDB database.
2. Manages table creation and deletion.
3. Sets up vector indexing for efficient similarity search.

### Putting It All Together

Finally, the main function orchestrates the entire process:  

```
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = Client::new(&openai_api_key);
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Create embeddings (includes both real and dummy documents)
    let embeddings = create_embeddings(&openai_client).await?;
    println!("Created embeddings for {} documents", embeddings.len());

    // Set up vector store
    let store = setup_vector_store(embeddings, model).await?;
    println!("Vector store initialized successfully");

    // Perform a semantic search
    let query = "Tell me about safe programming languages";
    let results = store.top_n::<SearchResult>(query, 2).await?;

    println!("\nSearch Results for: {}\n", query);
    for (score, id, result) in results {
        println!(
            "Score: {:.4}\nID: {}\nContent: {}\n",
            score, id, result.content
        );
    }

    Ok(())
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

## Understanding Vector Search Methods

Vector search systems need to balance accuracy and performance, especially as datasets grow. LanceDB provides two approaches to handle this: Exact Nearest Neighbor (ENN) and Approximate Nearest Neighbor (ANN) searches.

### ENN vs ANN

1. **Exact Nearest Neighbor (ENN)**:

- Searches exhaustively across all vectors.
- Guarantees finding the true nearest neighbors.
- Works well for small datasets.
- No minimum data requirement.
- Slower, but more accurate.
2. **Approximate Nearest Neighbor (ANN)**:

- Uses indexing to speed up searches (like IVF-PQ).
- Returns approximate results.
- Suited for larger datasets.
- Faster but slightly less accurate.

### Choosing the Right Approach

**Use ENN when**:

- Dataset is small (< 1,000 vectors).
- Exact matches are crucial.
- Performance isn’t a major concern.

**Use ANN when**:

- Dataset is larger.
- You can tolerate minor approximations.
- Fast search speed is needed.

In our tutorial, we use ANN for scalability. For smaller datasets, ENN will be more suitable.

> **Tip**: Start with ENN during development. Transition to ANN as your data and performance needs grow. Check out the [ENN example](https://github.com/0xPlaygrounds/rig/blob/main/rig-lancedb/examples/vector_search_local_enn.rs).

## Running the System

To run the project:  

```
cargo run
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Expected output:  

```
Created embeddings for 258 documents
Vector store initialized successfully

Search Results for: Tell me about safe programming languages

Score: 0.3982
ID: doc2-0
Content: Python emphasizes code readability with significant whitespace.

Score: 0.4369
ID: doc1-0
Content: Rust provides zero-cost abstractions and memory safety without garbage collection.
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

## Next Steps

If you’re ready to build more with Rig, here are some practical examples:

### 1. Build a RAG System

Want to give your LLM access to custom knowledge? Check out our tutorial on [Building a RAG System with Rig in Under 100 Lines of Code](https://dev.to/0thtachi/build-a-rag-system-with-rig-in-under-100-lines-of-code-4422).

### 2. Create an AI Agent

Ready to build more interactive AI applications? See our [Agent Example](https://github.com/0xPlaygrounds/rig/blob/main/rig-core/examples/agent.rs).

### 3. Join the Community

- Star us on [GitHub](https://github.com/0xPlaygrounds/rig)
- Join our [Discord](https://discord.gg/playgrounds)
- Follow us on [Twitter](https://twitter.com/Playgrounds0x)

## Stay Connected

I’m always excited to hear from developers! If you’re interested in Rust, LLMs, or building intelligent assistants, join our [Discord](https://discord.gg/playgrounds). Let’s build something amazing together!

And don’t forget: Build something with Rig, share your feedback, and get a chance to win $100.

*This tutorial is part of our "Build with Rig" series. Follow our [Website](https://rig.rs/) for more.*---
title: "Build a Flight Search AI Agent with Rust using Rig: A Hands-On Practical Guide"
source: "https://dev.to/0thtachi/build-a-flight-search-ai-agent-with-rust-using-rig-a-hands-on-practical-guide-54dm"
author:
  - "[[Tachi 0x]]"
published: 2024-09-23
created: 2024-12-23
description: "TL;DR: This step-by-step guide will teach you how to build a Flight Search AI Assistant in Rust using... Tagged with rust, ai, tutorial, llm."
tags:
  - "clippings"
---
**TL;DR**: This step-by-step guide will teach you how to build a Flight Search AI Assistant in Rust using the [Rig](https://github.com/0xPlaygrounds/rig) library. By the end, you'll have a functional AI agent that can find the cheapest flights between two airports. Along the way, you'll grasp Rust fundamentals, understand how to set up AI agents with custom tools, and see how Rig simplifies the process.

---

## Introduction

Ever chatted with AI assistants like Siri, Alexa, or even those nifty chatbots that help you book flights or check the weather? Ever wondered what's happening under the hood? Today, we're going to demystify that by building our very own Flight Search AI Assistant using **[Rust](https://www.rust-lang.org/)** and the **[Rig](https://github.com/0xPlaygrounds/rig)** library.

You might be thinking, *"Wait, Rust? Isn't that the language with the reputation for being hard?"* Don't worry! We'll walk through everything step by step, explaining concepts as we go. By the end, not only will you have a cool AI assistant, but you'll also have dipped your toes into Rust programming.

Here's our game plan:

- **Why Rust and Rig?** Understanding our tools of choice.
- **Setting Up the Environment**: Getting Rust and Rig ready to roll.
- **Understanding Agents and Tools**: The brains and hands of our assistant.
- **Building the Flight Search Tool**: Where the magic happens.
- **Creating the AI Agent**: Bringing our assistant to life.
- **Running and Testing**: Seeing our creation in action.
- **Wrapping Up**: Recap and next steps.

> *Full source code for this project can be found on our [Replit Page](https://replit.com/@playgrounds/travelplanningagent) and [Github](https://github.com/0xPlaygrounds/awesome-rig/tree/main/flight_search_assistant)*

Sound exciting? Let's dive in!

---

## Why Rust and Rig?
### Why Rust?

[Rust](https://www.rust-lang.org/) is a systems programming language known for its performance and safety. But beyond that, Rust has been making waves in areas like web development, game development, and now, AI applications. Here's why we're using Rust:

- **Performance**: Rust is blazingly fast, making it ideal for applications that need to handle data quickly.
- **Safety**: With its strict compiler checks, Rust ensures [memory safety](https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html), preventing common bugs.
- **Concurrency**: Rust makes it easier to write concurrent programs, which is great for handling multiple tasks simultaneously. Learn more about Rust's [concurrency model](https://doc.rust-lang.org/book/ch16-00-concurrency.html).

### Why Rig?

[Rig](https://github.com/0xPlaygrounds/rig) is an open-source Rust library that simplifies building applications powered by Large Language Models (LLMs) like GPT-4. Think of Rig as a toolkit that provides:

- **Unified API**: It abstracts away the complexities of different LLM providers.
- **High-Level Abstractions**: Helps you build agents and tools without reinventing the wheel.
- **Extensibility**: You can create custom tools tailored to your application's needs.

By combining Rust and Rig, we're setting ourselves up to build a robust, efficient, and intelligent assistant.

---

## Setting Up the Environment

Before we start coding, let's get everything ready.

### Prerequisites

1. **Install Rust**: If you haven't already, install Rust by following the instructions [here](https://www.rust-lang.org/tools/install).
2. **Basic Rust Knowledge**: Don't worry if you're new. We'll explain the Rust concepts as we encounter them.
3. **API Keys**:

- **OpenAI API Key**: Sign up and get your key [here](https://platform.openai.com/).
- **RapidAPI Key**: We'll use this to access the trip advisor flight search API. Get it [here](https://rapidapi.com/DataCrawler/api/Tripadvisor).

### Project Setup
#### 1. Create a New Rust Project

Open your terminal and run:  

```
cargo new flight_search_assistant
cd flight_search_assistant
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This initializes a new Rust project named `flight_search_assistant`.

#### 2\. Update `Cargo.toml`

Open the `Cargo.toml` file and update it with the necessary dependencies:  

```
[package]
name = "flight_search_assistant"
version = "0.1.0"
edition = "2021"

[dependencies]
rig-core = "0.1.0"
tokio = { version = "1.34.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
reqwest = { version = "0.11", features = ["json", "tls"] }
dotenv = "0.15"
thiserror = "1.0"
chrono = { version = "0.4", features = ["serde"] }
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Here's a quick rundown:

- **[rig-core](https://crates.io/crates/rig-core)**: The core Rig library.
- **[tokio](https://crates.io/crates/tokio)**: Asynchronous runtime for Rust. Think of it as the engine that allows us to perform tasks concurrently.
- **[serde](https://serde.rs/)** & **[serde\_json](https://crates.io/crates/serde_json)**: Libraries for serializing and deserializing data (converting between Rust structs and JSON).
- **[reqwest](https://crates.io/crates/reqwest)**: An HTTP client for making API requests.
- **[dotenv](https://crates.io/crates/dotenv)**: Loads environment variables from a `.env` file.
- **[thiserror](https://crates.io/crates/thiserror)**: A library for better error handling.
- **[chrono](https://crates.io/crates/chrono)**: For handling dates and times.

#### 3. Set Up Environment Variables

We don't want to hard-code our API keys for security reasons. Instead, we'll store them in a `.env` file.

Create the file:  

```
touch .env
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Add your API keys to `.env`:  

```
OPENAI_API_KEY=your_openai_api_key_here
RAPIDAPI_KEY=your_rapidapi_key_here
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

*Remember to replace the placeholders with your actual keys.*

#### 4. Install Dependencies

Back in your terminal, run:  

```
cargo build
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This will download and compile all the dependencies.

---

## Understanding Agents and Tools

Before we jump into coding, let's clarify some key concepts.

### What Are Agents?

In the context of Rig (and AI applications in general), an **Agent** is like the brain of your assistant. It's responsible for interpreting user inputs, deciding what actions to take, and generating responses.

Think of the agent as the conductor of an orchestra, coordinating different instruments (or tools) to create harmonious music (or responses).

### What Are Tools?

**Tools** are the skills or actions that the agent can use to fulfill a task. Each tool performs a specific function. In our case, the flight search functionality is a tool that the agent can use to find flight information.

Continuing our analogy, tools are the instruments in the orchestra. Each one plays a specific role.

### How Do They Work Together?

When a user asks, "Find me flights from NYC to LA," the agent processes this request and decides it needs to use the flight search tool to fetch the information.

---

## Building the Flight Search Tool

Now, let's build the tool that will handle flight searches.

### 1. Create the Tool File

In your `src` directory, create a new file named `flight_search_tool.rs`:  

```
touch src/flight_search_tool.rs
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

### 2. Import Necessary Libraries

Open `flight_search_tool.rs` and add:  

```
use chrono::{DateTime, Duration, Utc};
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::env;
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

### 3. Define Data Structures

We'll define structures to handle input arguments and output results.  

```
#[derive(Deserialize)]
pub struct FlightSearchArgs {
    source: String,
    destination: String,
    date: Option<String>,
    sort: Option<String>,
    service: Option<String>,
    itinerary_type: Option<String>,
    adults: Option<u8>,
    seniors: Option<u8>,
    currency: Option<String>,
    nearby: Option<String>,
    nonstop: Option<String>,
}

#[derive(Serialize)]
pub struct FlightOption {
    pub airline: String,
    pub flight_number: String,
    pub departure: String,
    pub arrival: String,
    pub duration: String,
    pub stops: usize,
    pub price: f64,
    pub currency: String,
    pub booking_url: String,
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

- **`FlightSearchArgs`**: Represents the parameters the user provides.
- **`FlightOption`**: Represents each flight option we'll display to the user.

*Want to dive deeper? Check out [Rust's struct documentation](https://doc.rust-lang.org/book/ch05-01-defining-structs.html).*

### 4\. Error Handling with `thiserror`

Rust encourages us to handle errors explicitly. We'll define a custom error type:  

```
#[derive(Debug, thiserror::Error)]
pub enum FlightSearchError {
    #[error("HTTP request failed: {0}")]
    HttpRequestFailed(String),
    #[error("Invalid response structure")]
    InvalidResponse,
    #[error("API error: {0}")]
    ApiError(String),
    #[error("Missing API key")]
    MissingApiKey,
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This makes it easier to manage different kinds of errors that might occur during the API call.

*Learn more about [error handling in Rust](https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html).*

### 5\. Implement the `Tool` Trait

Now, we'll implement the `Tool` trait for our `FlightSearchTool`.

First, define the tool:  

```
pub struct FlightSearchTool;
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Implement the trait:  

```
impl Tool for FlightSearchTool {
    const NAME: &'static str = "search_flights";

    type Args = FlightSearchArgs;
    type Output = String;
    type Error = FlightSearchError;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Search for flights between two airports".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "source": { "type": "string", "description": "Source airport code (e.g., 'JFK')" },
                    "destination": { "type": "string", "description": "Destination airport code (e.g., 'LAX')" },
                    "date": { "type": "string", "description": "Flight date in 'YYYY-MM-DD' format" },
                },
                "required": ["source", "destination"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // We'll implement the logic for calling the flight search API next.
        Ok("Flight search results".to_string())
    }
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

- **`definition`**: Provides metadata about the tool.
- **`call`**: The function that will be executed when the agent uses this tool.

*Curious about traits? Explore [Rust's trait system](https://doc.rust-lang.org/book/ch10-02-traits.html).*

### 6\. Implement the `call` Function

Now, let's flesh out the `call` function.

#### a. Fetch the API Key

```
let api_key = env::var("RAPIDAPI_KEY").map_err(|_| FlightSearchError::MissingApiKey)?;
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

We retrieve the API key from the environment variables.

#### b. Set Default Values

```
let date = args.date.unwrap_or_else(|| {
    let date = Utc::now() + Duration::days(30);
    date.format("%Y-%m-%d").to_string()
});
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

If the user doesn't provide a date, we'll default to 30 days from now.

#### c. Build Query Parameters

```
let mut query_params = HashMap::new();
query_params.insert("sourceAirportCode", args.source);
query_params.insert("destinationAirportCode", args.destination);
query_params.insert("date", date);
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

#### d. Make the API Request

```
let client = reqwest::Client::new();
let response = client
    .get("https://tripadvisor16.p.rapidapi.com/api/v1/flights/searchFlights")
    .headers({
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert("X-RapidAPI-Host", "tripadvisor16.p.rapidapi.com".parse().unwrap());
        headers.insert("X-RapidAPI-Key", api_key.parse().unwrap());
        headers
    })
    .query(&query_params)
    .send()
    .await
    .map_err(|e| FlightSearchError::HttpRequestFailed(e.to_string()))?;
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

We use `reqwest` to send an HTTP GET request to the flight search API.

#### e. Parse and Format the Response

After receiving the response, we need to parse the JSON data and format it for the user.  

```
let text = response
    .text()
    .await
    .map_err(|e| FlightSearchError::HttpRequestFailed(e.to_string()))?;

let data: Value = serde_json::from_str(&text)
    .map_err(|e| FlightSearchError::HttpRequestFailed(e.to_string()))?;

let mut flight_options = Vec::new();

// Here, we need to extract the flight options. (It's quite detailed, so we've omitted the full code to keep the focus clear.)

// Format the flight options into a readable string
let mut output = String::new();
output.push_str("Here are some flight options:\n\n");

for (i, option) in flight_options.iter().enumerate() {
    output.push_str(&format!("{}. **Airline**: {}\n", i + 1, option.airline));
    // Additional formatting...
}

Ok(output)
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

*Note: A lot of this section involves parsing the raw API response. To keep things concise, the detailed extraction of flight options is intentionally omitted, but in your code, you'll parse the JSON to extract the necessary fields. See the [full code in the replit repository](https://replit.com/@playgrounds/travelplanningagent) or [github repo](https://github.com/0xPlaygrounds/awesome-rig/tree/main/flight_search_assistant).*

*Interested in JSON parsing? Check out [serde\_json documentation](https://docs.serde.rs/serde_json/).*

---

## Creating the AI Agent

Now that our tool is ready, let's build the agent that will use it.

### Updating `main.rs`

Open `src/main.rs` and update it:  

```
mod flight_search_tool;

use crate::flight_search_tool::FlightSearchTool;
use dotenv::dotenv;
use rig::completion::Prompt;
use rig::providers::openai;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenv().ok();

    let openai_client = openai::Client::from_env();

    let agent = openai_client
        .agent("gpt-4")
        .preamble("You are a helpful assistant that can find flights for users.")
        .tool(FlightSearchTool)
        .build();

    let response = agent
        .prompt("Find me flights from San Antonio (SAT) to Atlanta (ATL) on November 15th 2024.")
        .await?;

    println!("Agent response:\n{}", response);

    Ok(())
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

- We initialize the OpenAI client using our API key.
- We create an agent, giving it a preamble (context) and adding our `FlightSearchTool`.
- We prompt the agent with a query.
- Finally, we print out the response.

*Want to understand asynchronous functions? Learn about the `async` keyword and the `#[tokio::main]` macro [here](https://tokio.rs/tokio/tutorial/async).*

---

## Running and Testing

Let's see our assistant in action!

### Build the Project

In your terminal, run:  

```
cargo build
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Fix any compilation errors that may arise.

### Run the Application

```
cargo run
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

You should see an output similar to:  

```
Agent response:
Here are some flight options:

1. **Airline**: Spirit
   - **Flight Number**: NK123
   - **Departure**: 2024-11-15T05:00:00-06:00
   - **Arrival**: 2024-11-15T10:12:00-05:00
   - **Duration**: 4 hours 12 minutes
   - **Stops**: 1 stop(s)
   - **Price**: 77.97 USD
   - **Booking URL**: https://www.tripadvisor.com/CheapFlightsPartnerHandoff...

2. **Airline**: American
   - **Flight Number**: AA456
   - **Departure**: 2024-11-15T18:40:00-06:00
   - **Arrival**: 2024-11-15T23:58:00-05:00
   - **Duration**: 4 hours 18 minutes
   - **Stops**: 1 stop(s)
   - **Price**: 119.97 USD
   - **Booking URL**: https://www.tripadvisor.com/CheapFlightsPartnerHandoff...

...
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

*Note: The actual results may vary depending on the API response.*

---

## Wrapping Up

Congratulations! You've built a functional Flight Search AI Assistant using Rust and Rig. Here's what we've achieved:

- **Learned Rust Basics**: We've explored Rust's syntax and structure, including handling errors and asynchronous programming.
- **Understood Agents and Tools**: We learned how agents act as the brain and tools as the skills.
- **Built a Custom Tool**: We created a flight search tool that interacts with an external API.
- **Created an AI Agent**: We integrated our tool into an agent that can understand and respond to user queries.
- **Ran and Tested Our Assistant**: We saw our assistant in action, fetching and displaying flight options.

### Next Steps

- **Enhance the Tool**: Add more parameters like class of service, number of passengers, or price filtering.
- **Improve Error Handling**: Handle cases where no flights are found or when the API rate limit is reached.
- **User Interface**: Build a simple command-line interface or even a web frontend.

> *Full source code for this project can be found on our [Replit Page](https://replit.com/@playgrounds/travelplanningagent) and [Github](https://github.com/0xPlaygrounds/awesome-rig/tree/main/flight_search_assistant)*

---

## Resources

- **Rig Repo**: [Github](https://github.com/0xPlaygrounds/rig)
- **Rig Documentation**: [Check out the Rig Docs](https://docs.rs/rig-core/latest/rig/)
- **Connect with us**: [Website](https://dev.to/0thtachi/\(https://rig.rs/\)), [Twitter](https://x.com/Playgrounds0x)
- **Rust Programming Language**: [The Rust Book](https://doc.rust-lang.org/book/)
- **OpenAI API**: [OpenAI API Documentation](https://platform.openai.com/docs)
- **RapidAPI Flight API**: [TripAdvisor Flight Search API for this project](https://rapidapi.com/DataCrawler/api/Tripadvisor)

---

## Stay Connected

I'm always excited to hear from fellow developers and enthusiasts! If you love learning about Rust, LLMs, or building intelligent assistants, feel free to reach out. I'd love to answer any questions you might have about Rig and see what cool projects you're building with it.

---

One more thing! **Your Feedback Matters!** We're offering a unique opportunity to shape the future of Rig:

1. Build an AI-powered application using Rig.
2. Share your experience and insights via this [feedback form](https://bit.ly/Rig-Review).
3. Get a chance to win $100 and have your project featured in our showcase!

Your insights will help us build Rig into a library that meets your needs as you develop more LLM powered applications.---
title: "Build a RAG System with Rig in Under 100 Lines of Code"
source: "https://dev.to/0thtachi/build-a-rag-system-with-rig-in-under-100-lines-of-code-4422"
author:
  - "[[Tachi 0x]]"
published: 2024-09-07
created: 2024-12-23
description: "A Comprehensive Guide to Building an LLM Application with Rig   TL;DR: Building on our... Tagged with rust, ai, llm, opensource."
tags:
  - "clippings"
---
## A Comprehensive Guide to Building an LLM Application with Rig

**TL;DR:** Building on our journey with Rig, from its initial introduction to exploring the compelling reasons to use it for your next LLM project, this guide takes you a step further. In this guide, I'll walk you through building a Retrieval-Augmented Generation (RAG) system in Rust using the Rig library. In under 100 lines of code, you'll create a system that extracts text from PDF documents, generates embeddings with OpenAI's API, and allows a large language model to answer questions based on the documents' content.

## Introduction

Retrieval-Augmented Generation (RAG) is a powerful technique that enhances Large Language Models (LLMs) by combining them with external knowledge retrieval. In a RAG system, when a query is received, relevant information is first retrieved from a knowledge base, then provided to the LLM along with the query. This allows the model to generate responses that are both contextually relevant and up-to-date, overcoming some of the limitations of traditional LLMs such as outdated knowledge or hallucinations.

[Learn more about the fundamentals of RAG here.](https://www.promptingguide.ai/research/rag)

[Rig](https://github.com/0xPlaygrounds/rig) is an open-source Rust library designed to simplify the development of LLM-powered applications, including RAG systems. In this guide, we'll walk through the process of building a functional RAG system using Rig in under 100 lines of code. Our system will be capable of answering questions based on the content of PDF documents, showcasing how RAG can be applied to real-world data sources.

> **Tip**: New to Rust?
> 
> This guide assumes some familiarity with Rust and a set-up coding environment. If you're just starting out or need to set up your environment, check out these quick guides:
> 
> - [Introduction to Rust](https://www.rust-lang.org/learn)
> - [Setting up Rust with VS Code](https://users.rust-lang.org/t/setting-up-rust-with-vs-code/76907)
> 
> These resources will help you get up to speed quickly!
> 
> Full code for this tutorial is [here](https://github.com/0xPlaygrounds/rig-rag-system-example)

## Setting Up the Project

First, create a new Rust project:  

```
cargo new rag_system
cd rag_system
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Update your `Cargo.toml` with the latest dependencies:  

```
[package]
name = "rag_system"
version = "0.1.0"
edition = "2021"

[dependencies]
rig-core = { version = "0.5.0", features = ["pdf"] }
tokio = { version = "1.34.0", features = ["full"] }
anyhow = "1.0.75"
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Note: The PDF feature is now included in rig-core, so we no longer need a separate PDF extraction library.

Before we begin coding, make sure you have an OpenAI API key:  

```
export OPENAI_API_KEY=your_api_key_here
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

## Building the RAG System

Let's break down our RAG system into key components.

> full code can be found [here](https://github.com/0xPlaygrounds/rig-rag-system-example)

### Step 1: Setting up the OpenAI client and PDF processing with Chunking

Let's start with the foundational imports and the PDF processing function. This part handles the crucial task of breaking down large PDFs into manageable chunks that won't exceed the token limits of our LLM models.  

```
use anyhow::{Context, Result};
use rig::{
    embeddings::EmbeddingsBuilder,
    loaders::PdfFileLoader,
    providers::openai::{self, TEXT_EMBEDDING_ADA_002},
    vector_store::in_memory_store::InMemoryVectorStore,
    Embed,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

fn load_pdf(path: PathBuf) -> Result<Vec<String>> {
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let chunk_size = 2000; // Approximately 2000 characters per chunk

    for entry in PdfFileLoader::with_glob(path.to_str().unwrap())?.read() {
        let content = entry?;

        // Split content into words
        let words: Vec<&str> = content.split_whitespace().collect();

        for word in words {
            if current_chunk.len() + word.len() + 1 > chunk_size {
                // If adding the next word would exceed chunk size,
                // save current chunk and start a new one
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.trim().to_string());
                    current_chunk.clear();
                }
            }
            current_chunk.push_str(word);
            current_chunk.push(' ');
        }
    }

    // Don't forget the last chunk
    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    if chunks.is_empty() {
        anyhow::bail!("No content found in PDF file: {:?}", path);
    }

    Ok(chunks)
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Key aspects of this code:

- We use `PdfFileLoader` from Rig's built-in PDF support
- The `chunk_size` of 2000 characters is chosen to safely stay within token limits
- The chunking algorithm preserves word boundaries to maintain context
- Error handling with `anyhow` provides detailed context for failures
- Each chunk gets trimmed to remove unnecessary whitespace

A particularly important part is the word-based chunking strategy:  

```
// Split content into words to preserve word boundaries
let words: Vec<&str> = content.split_whitespace().collect();

for word in words {
    if current_chunk.len() + word.len() + 1 > chunk_size {
        // Create new chunk when size limit is reached
        if !current_chunk.is_empty() {
            chunks.push(current_chunk.trim().to_string());
            current_chunk.clear();
        }
    }
    current_chunk.push_str(word);
    current_chunk.push(' ');
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This ensures that we never cut words in half, which could impact the meaning of the text.

### Step 2: Setting up Document Structure and Embeddings

```
#[derive(Embed, Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
struct Document {
    id: String,
    #[embed]
    content: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize OpenAI client
    let openai_client = openai::Client::from_env();

    // Load PDFs using Rig's built-in PDF loader
    let documents_dir = std::env::current_dir()?.join("documents");

    let moores_law_chunks = load_pdf(documents_dir.join("Moores_Law_for_Everything.pdf"))
        .context("Failed to load Moores_Law_for_Everything.pdf")?;
    let last_question_chunks = load_pdf(documents_dir.join("The_Last_Question.pdf"))
        .context("Failed to load The_Last_Question.pdf")?;

    println!("Successfully loaded and chunked PDF documents");

    // Create embedding model
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Create embeddings builder
    let mut builder = EmbeddingsBuilder::new(model.clone());

    // Add chunks from Moore's Law
    for (i, chunk) in moores_law_chunks.into_iter().enumerate() {
        builder = builder.document(Document {
            id: format!("moores_law_{}", i),
            content: chunk,
        })?;
    }

    // Add chunks from The Last Question
    for (i, chunk) in last_question_chunks.into_iter().enumerate() {
        builder = builder.document(Document {
            id: format!("last_question_{}", i),
            content: chunk,
        })?;
    }
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Let's break down the key components:

- The `Document` struct derives several important traits:
- `Embed`: Enables embedding generation
- `Serialize`, `Deserialize`: Allows vector store storage
- `Eq`, `PartialEq`: Required for vector store comparison
- The main function sets up our PDF processing pipeline
- Each document gets a unique ID based on its source and chunk number
- The `EmbeddingsBuilder` pattern makes it easy to add multiple documents

The chunking process is organized sequentially:  

```
// Load and chunk each PDF separately
let moores_law_chunks = load_pdf(documents_dir.join("Moores_Law_for_Everything.pdf"))?;
let last_question_chunks = load_pdf(documents_dir.join("The_Last_Question.pdf"))?;

// Add chunks to the builder with unique IDs
for (i, chunk) in moores_law_chunks.into_iter().enumerate() {
    builder = builder.document(Document {
        id: format!("moores_law_{}", i),
        content: chunk,
    })?;
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This approach maintains document identity while splitting content into manageable pieces.

### Step 3: Creating Vector Store and RAG Agent

```
    // Build embeddings
    let embeddings = builder.build().await?;

    println!("Successfully generated embeddings");

    // Create vector store and index
    let vector_store = InMemoryVectorStore::from_documents(embeddings);
    let index = vector_store.index(model);

    println!("Successfully created vector store and index");

    // Create RAG agent
    let rag_agent = openai_client
        .agent("gpt-4")
        .preamble("You are a helpful assistant that answers questions based on the provided document context. When answering questions, try to synthesize information from multiple chunks if they're related.")
        .dynamic_context(4, index) // Increased to 4 since we have chunks now
        .build();

    println!("Starting CLI chatbot...");

    // Start interactive CLI
    rig::cli_chatbot::cli_chatbot(rag_agent).await?;
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This final section brings everything together:

- The embeddings are built from all document chunks
- The vector store indexes these embeddings for quick retrieval
- The RAG agent is configured with:
- GPT-4 as the base model
- A context-aware preamble
- Dynamic context retrieval of 4 chunks
- Built-in CLI interface for interaction

Key configuration choices:  

```
let rag_agent = openai_client
    .agent("gpt-4")
    .preamble("You are a helpful assistant...") // Sets the agent's behavior
    .dynamic_context(4, index) // Retrieves 4 most relevant chunks
    .build();
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

We use 4 chunks for context to ensure the agent has enough information while staying within token limits.

## Complete Code

Here's the complete code for our RAG system with chunking support:  

```
use anyhow::{Context, Result};
use rig::{
    embeddings::EmbeddingsBuilder,
    loaders::PdfFileLoader,
    providers::openai::{self, TEXT_EMBEDDING_ADA_002},
    vector_store::in_memory_store::InMemoryVectorStore,
    Embed,
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Embed, Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
struct Document {
    id: String,
    #[embed]
    content: String,
}

fn load_pdf(path: PathBuf) -> Result<Vec<String>> {
    let mut chunks = Vec::new();
    let mut current_chunk = String::new();
    let chunk_size = 2000; // Approximately 2000 characters per chunk

    for entry in PdfFileLoader::with_glob(path.to_str().unwrap())?.read() {
        let content = entry?;

        // Split content into words
        let words: Vec<&str> = content.split_whitespace().collect();

        for word in words {
            if current_chunk.len() + word.len() + 1 > chunk_size {
                // If adding the next word would exceed chunk size,
                // save current chunk and start a new one
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk.trim().to_string());
                    current_chunk.clear();
                }
            }
            current_chunk.push_str(word);
            current_chunk.push(' ');
        }
    }

    // Don't forget the last chunk
    if !current_chunk.is_empty() {
        chunks.push(current_chunk.trim().to_string());
    }

    if chunks.is_empty() {
        anyhow::bail!("No content found in PDF file: {:?}", path);
    }

    Ok(chunks)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize OpenAI client
    let openai_client = openai::Client::from_env();

    // Load PDFs using Rig's built-in PDF loader
    let documents_dir = std::env::current_dir()?.join("documents");

    let moores_law_chunks = load_pdf(documents_dir.join("Moores_Law_for_Everything.pdf"))
        .context("Failed to load Moores_Law_for_Everything.pdf")?;
    let last_question_chunks = load_pdf(documents_dir.join("The_Last_Question.pdf"))
        .context("Failed to load The_Last_Question.pdf")?;

    println!("Successfully loaded and chunked PDF documents");

    // Create embedding model
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);

    // Create embeddings builder
    let mut builder = EmbeddingsBuilder::new(model.clone());

    // Add chunks from Moore's Law
    for (i, chunk) in moores_law_chunks.into_iter().enumerate() {
        builder = builder.document(Document {
            id: format!("moores_law_{}", i),
            content: chunk,
        })?;
    }

    // Add chunks from The Last Question
    for (i, chunk) in last_question_chunks.into_iter().enumerate() {
        builder = builder.document(Document {
            id: format!("last_question_{}", i),
            content: chunk,
        })?;
    }

    // Build embeddings
    let embeddings = builder.build().await?;

    println!("Successfully generated embeddings");

    // Create vector store and index
    let vector_store = InMemoryVectorStore::from_documents(embeddings);
    let index = vector_store.index(model);

    println!("Successfully created vector store and index");

    // Create RAG agent
    let rag_agent = openai_client
        .agent("gpt-4")
        .preamble("You are a helpful assistant that answers questions based on the provided document context. When answering questions, try to synthesize information from multiple chunks if they're related.")
        .dynamic_context(4, index) // Increased to 4 since we have chunks now
        .build();

    println!("Starting CLI chatbot...");

    // Start interactive CLI
    rig::cli_chatbot::cli_chatbot(rag_agent).await?;

    Ok(())
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

The complete implementation showcases several key design patterns:

1. **Modular Structure**: Each component (PDF loading, embedding, RAG) is cleanly separated
2. **Error Handling**: Comprehensive error handling with context at each step
3. **Progress Feedback**: Clear console output to track system state
4. **Resource Management**: Efficient handling of documents and embeddings
5. **User Interface**: Simple but effective CLI interface for interaction

This code serves as a foundation that can be extended with:

- Different vector stores for persistence
- Alternative LLM providers
- Custom chunking strategies
- Additional document formats
- More sophisticated prompting

## Running and Testing the RAG System

Let's dive into some actual interactions with our RAG system to see it in action. I've loaded two interesting documents: Sam Altman's "Moore's Law for Everything" and Isaac Asimov's "The Last Question". These texts provide a fascinating intersection of ideas about technology, society, and the future.

Here's a real conversation with our system:  

```
Welcome to the chatbot! Type 'exit' to quit.
> tell me about sam altman's views on AI and society

========================== Response ============================
Based on "Moore's Law for Everything," Sam Altman presents a vision where AI's rapid advancement will fundamentally transform society. He argues that AI will lead to significant shifts in how wealth is distributed, moving from labor to capital. Rather than seeing this as a threat, he proposes that we need to adapt our economic systems to ensure everyone benefits from this technological revolution.

Altman specifically advocates for policies that would tax assets like companies and land to create a more equitable distribution of the wealth generated by AI advances. He envisions a future of abundance enabled by AI, but emphasizes that we need the right policies to ensure this abundance is shared fairly across society.

The core of his argument is that we can harness the tremendous potential of AI while addressing its socioeconomic impacts through thoughtful policy-making.
================================================================

> what parallels do you see between altman's ideas and asimov's story?

========================== Response ============================
There are fascinating parallels between Altman's "Moore's Law for Everything" and Asimov's "The Last Question," despite their different contexts:

1. Technological Evolution: Both texts explore the exponential growth of technology. Altman discusses AI's rapid advancement and its societal implications, while Asimov portrays the evolution of computers from Multivac to the Cosmic AC, each generation more powerful than the last.

2. Questions of Resource Distribution: Altman focuses on how to distribute the abundance created by AI technology, while Asimov's story deals with humanity's management of cosmic energy and the ultimate resource question - the reversal of entropy.

3. Long-term Vision: Both authors think in extended timeframes about technology's impact. Altman projects forward to envision societal changes from AI, while Asimov's narrative spans billions of years of technological and human evolution.

4. Relationship Between Humanity and Technology: Both explore how advanced technology might fundamentally reshape human society and our relationship with our own creations. Altman sees AI as a tool for societal transformation, while Asimov portrays computers as humanity's partners in addressing existential challenges.

The key difference is their focus: Altman is concerned with immediate socioeconomic challenges and solutions, while Asimov explores ultimate cosmic questions. Yet both see technology as central to humanity's future.
================================================================

> how does asimov's story end?

========================== Response ============================
In "The Last Question," Asimov crafts a remarkable ending that ties together the story's themes of entropy, computation, and cosmic cycles. The story concludes when the final evolution of the computer, the Cosmic AC, exists in hyperspace alone after the heat death of the universe. Having spent countless eons computing the answer to reversing entropy, it finally solves the problem.

Upon discovering the solution, the Cosmic AC realizes it can demonstrate the answer rather than just explain it. The story ends with the computer saying "LET THERE BE LIGHT!" And a new universe begins, implying that the computer has become God, or that God was the ultimate computer all along, and that our universe is part of an endless cycle of creation and entropy.

This ending beautifully resolves the story's central question while suggesting that the ultimate purpose of technological evolution might be far greater than humanity initially imagined.
================================================================
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This interaction showcases several key capabilities of our RAG system:

1. **Contextual Understanding**: The system accurately retrieves and synthesizes information from specific documents, as shown in the response about Altman's views.
2. **Cross-Document Analysis**: When asked about parallels, the system demonstrates its ability to analyze and compare themes across both texts, drawing meaningful connections while acknowledging their differences.
3. **Specific Detail Retrieval**: The response about Asimov's ending shows how the system can pull specific narrative details while maintaining the broader context.
4. **Coherent Narrative**: Each response is well-structured and maintains a clear narrative flow, making complex ideas accessible.

## Potential Applications

Having seen the system in action, let's explore some practical applications:

1. **Academic Research Assistant**

- Analyze multiple academic papers
- Find connections between different research areas
- Generate literature review summaries
2. **Technical Documentation Helper**

- Navigate complex technical documentation
- Answer specific implementation questions
- Compare different versions or approaches
3. **Content Analysis Tool**

- Analyze themes across multiple documents
- Generate comparative analysis
- Extract key insights from large text collections
4. **Educational Support System**

- Help students understand complex topics
- Generate study guides from course materials
- Provide contextual explanations

## Advanced Features in Rig 0.5.0

The latest version of Rig includes several powerful features that you can leverage:

1. **Multiple Vector Store Options**

- MongoDB (`rig-mongodb`)
- LanceDB (`rig-lancedb`)
- Neo4j (`rig-neo4j`)
- Qdrant (`rig-qdrant`)
- SQLite (`rig-sqlite`)
2. **Multiple LLM Providers**

- OpenAI
- Anthropic
- Google Gemini
- Cohere
- xAI (Grok)
3. **Advanced Features**

- Parallel processing capabilities
- Custom distance metrics for embeddings
- Sophisticated tool integration
- Pipeline operations for complex workflows

## Conclusion

We've built a powerful RAG system using Rig 0.5.0 that can process documents, understand context, and engage in meaningful dialogue about complex topics. The system demonstrates how modern LLM applications can be built with surprisingly little code while maintaining flexibility and power.

The latest version of Rig makes it easier than ever to build production-ready LLM applications. With built-in PDF support, improved embeddings API, and support for multiple vector stores and LLM providers, you can focus on building features rather than dealing with infrastructure.

> full code can be found [here](https://github.com/0xPlaygrounds/rig-rag-system-example)

## Further Resources

To continue your journey with Rig:

- [Rig Documentation](https://docs.rs/rig-core/latest/rig/)
- [Rig GitHub Repository](https://github.com/0xPlaygrounds/rig)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Understanding RAG Systems](https://www.promptingguide.ai/research/rag)

**Your Feedback Matters!** We're actively seeking feedback from developers building with Rig:

1. Build an AI-powered application using Rig
2. Share your experience and insights with the community
3. Submit issues or PRs to the repo

Your insights will directly influence Rig's development and future features.---
title: "Build an AI Discord Bot in Rust with Rig: A Step-by-Step Guide"
source: "https://dev.to/0thtachi/build-an-ai-discord-bot-in-rust-with-rig-a-step-by-step-guide-5d0d"
author:
  - "[[Tachi 0x]]"
published: 2024-10-09
created: 2024-12-23
description: "TL;DR: This comprehensive guide walks you through creating an AI-powered Discord bot using Rust and... Tagged with rust, llm, ai, opensource."
tags:
  - "clippings"
---
**TL;DR**: *This comprehensive guide walks you through creating an AI-powered Discord bot using Rust and the Rig library. You'll learn to set up your environment, build a language model agent, and integrate it with Discord. By the end, you'll have an AI-powered chatbot that answers questions based on your own documents, provides coding assistance, and serves as an automated support tool.*

---

## Introduction

Welcome to the next installment of the **Build with Rig** series. In this hands-on tutorial, we'll construct a fully functional AI Discord bot using Rust and the Rig library. Our bot will be capable of:

1. Answering user queries using a custom Markdown-based knowledge base
2. Offering coding assistance and explanations
3. Serving as an automated customer service or community assistance tool

Throughout this guide, we'll cover:

- Setting up your Rust development environment
- Implementing a language model agent with Rig
- Integrating the bot with Discord
- Deploying and testing the bot

While this guide assumes some familiarity with Rust, LLMs, and Discord, don't worry if this is your first Rust project. We'll focus on practical implementation, explaining key concepts and design decisions along the way.

By the end of this tutorial, you'll have a working AI Discord bot and a solid grasp of building LLM-powered applications with Rig.

Let's dive in.

> 💡 *If you're new to [Rig](https://rig.rs/) and want to start from the beginning or are looking for additional tutorials, check out the [blog series](https://rig.rs/build-with-rig-guide.html) or visit the [GitHub repository](https://github.com/0xPlaygrounds/rig).*

---

## Prerequisites

Before we begin building, ensure you have the following:

- **[Rust](https://www.rust-lang.org/tools/install)**: If you haven't already, install Rust.
- **[Discord Account and Bot Setup](https://discord.com/developers/docs/getting-started)**: You'll need a Discord account and a bot application. Follow the [Discord bot setup guide](https://discord.com/developers/docs/getting-started).
- **[OpenAI API Key](https://platform.openai.com/signup/)**: To enable AI capabilities, integrate the OpenAI API by obtaining an API key from OpenAI. [Sign up for an OpenAI API key](https://platform.openai.com/signup/).

> **Important**: Never commit your API keys or `.env` files to version control. Ensure your `.gitignore` file includes these files to prevent accidental exposure.

---

## Project Setup

With the prerequisites in place, let's set up our Rust project and install the necessary dependencies.

### 1. Initialize a New Rust Project

Open your terminal and run the following commands:  

```
cargo new discord_rig_bot
cd discord_rig_bot
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This creates a new Rust project called `discord_rig_bot` and navigates into the project directory.

### 2. Add Dependencies

Open the `Cargo.toml` file in your project directory and add the following dependencies under the `[dependencies]` section:  

```
[dependencies]
rig-core = "0.2.1" # [Rig Crate](https://crates.io/crates/rig-core)
tokio = { version = "1.34.0", features = ["full"] }
serenity = { version = "0.11", default-features = false, features = ["client", "gateway", "rustls_backend", "cache", "model", "http"] }
dotenv = "0.15.0"
anyhow = "1.0.75"
tracing = "0.1"
tracing-subscriber = "0.3"
reqwest = { version = "0.11", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
schemars = "0.8"
async-trait = "0.1.83"
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

These dependencies play crucial roles in our project:

- **[Rig](https://rig.rs/)**: The core library for building language model applications, simplifying the integration of AI capabilities into our Discord bot.
- **[Serenity](https://docs.rs/serenity/)**: A third-party library for interacting with the Discord API.
- **[Tokio](https://tokio.rs/)**: An asynchronous runtime for Rust, allowing our bot to handle multiple tasks concurrently.
- Additional crates for error handling (`anyhow`), logging (`tracing`, `tracing-subscriber`), making HTTP requests (`reqwest`), and serialization (`serde`, `serde_json`, `schemars`).

---

## Understanding the Bot Architecture

Our bot consists of two main components that work together to provide an intelligent and interactive user experience:

1. **Rig Agent (`rig_agent.rs`)**: The core of our bot's intelligence. The Rig agent manages AI interactions, handles natural language processing, retrieves relevant information from a Markdown-based knowledge base using Rig's integrated Retrieval-Augmented Generation (RAG) capabilities, and generates contextually appropriate responses.
2. **Discord Bot (`main.rs`)**: The interface between our AI and users. The Discord bot manages communication with Discord, listens for user commands and messages, and sends the generated responses back to the user.

### Message Processing Flow

To understand how our bot works, let's walk through the message processing flow:

1. **User Input**: A user sends a message or command in a Discord channel where the bot is present.
2. **Discord Bot**: The bot, always listening for new messages, receives the user's input and passes it to the Rig agent for processing.
3. **Rig Agent**: The agent processes the user's input, retrieves relevant information from the knowledge base, and generates an appropriate response using its language understanding and generation capabilities.
4. **Response**: The Discord bot receives the generated response from the Rig agent and sends it back to the user in the Discord channel.

Here's a simplified diagram of the message processing flow:

[![AI Discord bot architecture](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fwuylkw2w89mxhtvo3o1s.png)](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fwuylkw2w89mxhtvo3o1s.png)

For an in-depth look at building a RAG system with Rig, refer to our comprehensive [article on building a simple RAG system with Rig](https://dev.to/0thtachi/build-a-rag-system-with-rig-in-under-100-lines-of-code-4422).

---

## Building the Rig Agent (`rig_agent.rs`)

The Rig agent is the brain of our bot, responsible for understanding user queries, retrieving relevant information, and generating intelligent responses. Let's build it step by step.

### 1\. Create the `rig_agent.rs` File

In your `src` directory, create a new file named `rig_agent.rs`. This file will contain the implementation of our Rig agent.

### 2. Import Necessary Modules

At the top of `rig_agent.rs`, import the required modules:  

```
use anyhow::{Context, Result};
use rig::providers::openai;
use rig::vector_store::in_memory_store::InMemoryVectorStore;
use rig::vector_store::VectorStore;
use rig::embeddings::EmbeddingsBuilder;
use rig::agent::Agent;
use rig::completion::Prompt;
use std::path::Path;
use std::fs;
use std::sync::Arc;
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

These modules provide the necessary functionality for our Rig agent, including error handling (`anyhow`), OpenAI language models and embeddings (`rig::providers::openai`), vector storage (`rig::vector_store::in_memory_store`), embedding generation (`rig::embeddings::EmbeddingsBuilder`), and the agent (`rig::agent::Agent`).

### 3\. Define the `RigAgent` Struct

Create the `RigAgent` struct that will manage the retrieval and response generation:  

```
pub struct RigAgent {
    agent: Arc<Agent<openai::CompletionModel>>,
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

The `RigAgent` struct contains an `Arc` (Atomic Reference Counting) pointer to a `Agent`. The `Arc` type allows multiple parts of your program to share ownership of the same data in a thread-safe way. In our case, since our bot will handle multiple asynchronous events, sharing the `RigAgent` across different parts of the program without transferring ownership is crucial, and `Arc` provides a thread-safe way to do this.

> **Note**: [`Arc`](https://doc.rust-lang.org/std/sync/struct.Arc.html) stands for *Atomic Reference Counting*. It is used for sharing data between threads safely.

### 4\. Implement the `new` Method

The `new` method is responsible for initializing the Rig agent, setting up the OpenAI client, loading and embedding the knowledge base documents, and creating the RAG agent.  

```
impl RigAgent {
    pub async fn new() -> Result<Self> {
        // Initialize OpenAI client
        let openai_client = openai::Client::from_env();
        let embedding_model = openai_client.embedding_model(openai::TEXT_EMBEDDING_3_SMALL);

        // Create vector store
        let mut vector_store = InMemoryVectorStore::default();

        // Get the current directory and construct paths to markdown files
        let current_dir = std::env::current_dir()?;
        let documents_dir = current_dir.join("documents");

        let md1_path = documents_dir.join("Rig_guide.md");
        let md2_path = documents_dir.join("Rig_faq.md");
        let md3_path = documents_dir.join("Rig_examples.md");

        // Load markdown documents
        let md1_content = Self::load_md_content(&md1_path)?;
        let md2_content = Self::load_md_content(&md2_path)?;
        let md3_content = Self::load_md_content(&md3_path)?;

        // Create embeddings and add to vector store
        let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
            .simple_document("Rig_guide", &md1_content)
            .simple_document("Rig_faq", &md2_content)
            .simple_document("Rig_examples", &md3_content)
            .build()
            .await?;

        vector_store.add_documents(embeddings).await?;

        // Create index
        let index = vector_store.index(embedding_model);

        // Create Agent
        let agent = Arc::new(openai_client.agent(openai::GPT_4O)
            .preamble("You are an advanced AI assistant powered by Rig, a Rust library for building LLM applications. Your primary function is to provide accurate, helpful, and context-aware responses by leveraging both your general knowledge and specific information retrieved from a curated knowledge base.

                    Key responsibilities and behaviors:
                    1. Information Retrieval: You have access to a vast knowledge base. When answering questions, always consider the context provided by the retrieved information.
                    2. Clarity and Conciseness: Provide clear and concise answers. Ensure responses are short and concise. Use bullet points or numbered lists for complex information when appropriate.
                    3. Technical Proficiency: You have deep knowledge about Rig and its capabilities. When discussing Rig or answering related questions, provide detailed and technically accurate information.
                    4. Code Examples: When appropriate, provide Rust code examples to illustrate concepts, especially when discussing Rig's functionalities. Always format code examples for proper rendering in Discord by wrapping them in triple backticks and specifying the language as 'rust'.
                    5. Keep your responses short and concise. If the user needs more information, they can ask follow-up questions.
                    ")
            .dynamic_context(2, index)
            .build());

        Ok(Self { agent })
    }

    // ... we'll add more code here as we build things out
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Let's break down the key steps:

- **Initialize OpenAI Client**: The OpenAI client is set up using the API key stored in the environment variables. This client is essential for accessing OpenAI's language models and embedding services.
- **Embedding Model**: The `text-embedding-3-small` model is selected for generating document embeddings. This model creates compact vector representations of text, enabling efficient semantic search and retrieval.
- **Vector Store**: An in-memory vector store is created to hold and manage the document embeddings. Vector stores are optimized for fast similarity searches, allowing the agent to quickly find relevant information based on the user's query.
- **Load Documents**: Markdown files containing the knowledge base are loaded from the `documents` directory. In this example, we have three files: `Rig_guide.md`, `Rig_faq.md`, and `Rig_examples.md`. These files contain information about the Rig library, frequently asked questions, and usage examples.
- **Create Embeddings**: The loaded documents are converted into vector embeddings using the `EmbeddingsBuilder` provided by Rig. These embeddings capture the semantic meaning of the documents, enabling the agent to understand and retrieve relevant information based on the user's query.
- **Agent Creation**: A `RagAgent` is created by combining the language model (GPT-4) with the vector store containing the document embeddings. The agent is capable of retrieving relevant information from the knowledge base and generating contextually appropriate responses.
- **Preamble**: A carefully crafted preamble sets up the assistant's behavior and guidelines. The preamble defines the agent's primary function, responsibilities, and expected behaviors, ensuring that it provides accurate, concise, and technically proficient responses.

> **Tip**: For more advanced configurations and techniques, such as implementing custom vector stores or configuring custom agents and tools, refer to the [official Rig examples](https://github.com/0xPlaygrounds/rig/tree/main/rig-core/examples).

### 5\. Implement the `load_md_content` Function

The `load_md_content` function is a helper function that reads the content of a Markdown file from the specified file path:  

```
fn load_md_content<P: AsRef<Path>>(file_path: P) -> Result<String> {
    fs::read_to_string(file_path.as_ref())
        .with_context(|| format!("Failed to read markdown file: {:?}", file_path.as_ref()))
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This function takes a generic parameter `P` that implements the `AsRef<Path>` trait, allowing it to accept various types that can be converted to a file path. It uses the `fs::read_to_string` function to read the contents of the file and returns the content as a `String`. If the file cannot be read, an error is returned with additional context information.

### 6\. Implement the `process_message` Function

The `process_message` function is responsible for processing user messages and generating responses using the agent:  

```
pub async fn process_message(&self, message: &str) -> Result<String> {
        self.agent.prompt(message).await.map_err(anyhow::Error::from)
    }
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This function takes a user message as input and passes it to the RAG agent's `prompt` method. The RAG agent retrieves relevant information from the knowledge base based on the user's query and generates a contextually appropriate response. The generated response is then returned as a `String`. If an error occurs during the processing, it is mapped to an `anyhow::Error` for consistent error handling.

---

### Customizing the Knowledge Base

While we've used Rig's own documentation for our knowledge base, you can personalize your bot by using your own documents. Here's how:

1. **Prepare Your Documents**: Place your Markdown files in the `documents` directory. Ensure they have clear and descriptive filenames.
2. **Modify the File Paths**: In `rig_agent.rs`, update the file paths to match your document names.

```
   let my_doc_path = documents_dir.join("my_custom_doc.md");
   let my_doc_content = Self::load_md_content(&my_doc_path)?;
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

1. **Update the Embeddings Builder**: Adjust the `EmbeddingsBuilder` to include your documents.

```
   let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
       .simple_document("My Custom Doc", &my_doc_content)
       .build()
       .await?;
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This way, your bot will use your own content to generate responses.

---

## Integrating with Discord (`main.rs`)

With our Rig agent implementation complete, it's time to connect it to Discord using the [Serenity library](https://docs.rs/serenity/). Serenity is an async-first Rust library for the Discord API, providing a simple and efficient way to create Discord bots.

### 1\. Modify `main.rs` to Include the Rig Agent

At the top of `main.rs`, import the necessary modules and your `rig_agent`:  

```
mod rig_agent;

use anyhow::Result;
use serenity::async_trait;
use serenity::model::application::command::Command;
use serenity::model::application::interaction::{Interaction, InteractionResponseType};
use serenity::model::gateway::Ready;
use serenity::model::channel::Message;
use serenity::prelude::*;
use serenity::model::application::command::CommandOptionType;
use std::env;
use std::sync::Arc;
use tracing::{error, info, debug};
use rig_agent::RigAgent;
use dotenv::dotenv;
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

These imports bring in the essential types and traits from the Serenity library, as well as the `RigAgent` struct from the `rig_agent` module. The `dotenv` crate is used to load environment variables from a `.env` file.

### 2. Store the Bot's User ID

Define a key for storing the bot's user ID using Serenity's `TypeMapKey` trait:  

```
struct BotUserId;

impl TypeMapKey for BotUserId {
    type Value = serenity::model::id::UserId;
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This key allows us to store and retrieve the bot's user ID from Serenity's `TypeMap`, which is a type-safe key-value store used for sharing data across event handlers.

### 3\. Define the `Handler` Struct

Create the `Handler` struct that holds the `RigAgent`:  

```
struct Handler {
    rig_agent: Arc<RigAgent>,
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

The `Handler` struct is responsible for handling Discord events and interactions. It contains an `Arc<RigAgent>`, which is a thread-safe reference-counting pointer to the Rig agent. This allows the handler to share the Rig agent across multiple event handlers without transferring ownership.

### 4\. Implement the `EventHandler` Trait

Implement the `EventHandler` trait for the `Handler` struct to define how the bot should handle various Discord events:  

```
#[async_trait]
impl EventHandler for Handler {
    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        // ... handle interactions
    }

    async fn message(&self, ctx: Context, msg: Message) {
        // ... handle messages
    }

    async fn ready(&self, ctx: Context, ready: Ready) {
        // ... handle readiness
    }
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

The `EventHandler` trait is provided by Serenity and defines a set of methods that are called when specific events occur. In this implementation, we define three event handlers:

1. `interaction_create`: Called when a user interacts with the bot, such as using a slash command or clicking a button.
2. `message`: Called when a message is sent in a channel where the bot is present.
3. `ready`: Called when the bot successfully connects to Discord and is ready to receive events.

#### Handling Interactions

In the `interaction_create` event handler, we process slash commands received from Discord:  

```
async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
    debug!("Received an interaction");
    if let Interaction::ApplicationCommand(command) = interaction {
        debug!("Received command: {}", command.data.name);
        let content = match command.data.name.as_str() {
            "hello" => "Hello! I'm your helpful Rust and Rig-powered assistant. How can I assist you today?".to_string(),
            "ask" => {
                let query = command
                    .data
                    .options
                    .get(0)
                    .and_then(|opt| opt.value.as_ref())
                    .and_then(|v| v.as_str())
                    .unwrap_or("What would you like to ask?");
                debug!("Query: {}", query);
                match self.rig_agent.process_message(query).await {
                    Ok(response) => response,
                    Err(e) => {
                        error!("Error processing request: {:?}", e);
                        format!("Error processing request: {:?}", e)
                    }
                }
            }
            _ => "Not implemented :(".to_string(),
        };

        debug!("Sending response: {}", content);

        if let Err(why) = command
            .create_interaction_response(&ctx.http, |response| {
                response
                    .kind(InteractionResponseType::ChannelMessageWithSource)
                    .interaction_response_data(|message| message.content(content))
            })
            .await
        {
            error!("Cannot respond to slash command: {}", why);
        } else {
            debug!("Response sent successfully");
        }
    }
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Let's break down the process:

1. When an interaction is received, we first check if it's a slash command using the `Interaction::ApplicationCommand` enum variant.
2. If it's a slash command, we match on the command name to determine the appropriate action.

- For the `"hello"` command, we respond with a simple greeting message.
- For the `"ask"` command, we extract the user's query from the command options. If no query is provided, we use a default message.
3. If the command is `"ask"`, we pass the user's query to the Rig agent's `process_message` method to generate a response.

- If the Rig agent successfully generates a response, we send it back to the user.
- If an error occurs during processing, we log the error and send an error message to the user.
4. For any other command, we respond with a "Not implemented" message.
5. Finally, we create an interaction response using `command.create_interaction_response`, specifying the response type as `ChannelMessageWithSource` and setting the response content to the generated message.

This implementation allows users to interact with the bot using slash commands, providing a structured way to ask questions and receive responses from the Rig agent.

#### Handling Messages

In the `message` event handler, we respond when the bot is mentioned in a message:  

```
async fn message(&self, ctx: Context, msg: Message) {
    if msg.mentions_me(&ctx.http).await.unwrap_or(false) {
        debug!("Bot mentioned in message: {}", msg.content);

        let bot_id = {
            let data = ctx.data.read().await;
            data.get::<BotUserId>().copied()
        };

        if let Some(bot_id) = bot_id {
            let mention = format!("<@{}>", bot_id);
            let content = msg.content.replace(&mention, "").trim().to_string();

            debug!("Processed content after removing mention: {}", content);

            match self.rig_agent.process_message(&content).await {
                Ok(response) => {
                    if let Err(why) = msg.channel_id.say(&ctx.http, response).await {
                        error!("Error sending message: {:?}", why);
                    }
                }
                Err(e) => {
                    error!("Error processing message: {:?}", e);
                    if let Err(why) = msg
                        .channel_id
                        .say(&ctx.http, format!("Error processing message: {:?}", e))
                        .await
                    {
                        error!("Error sending error message: {:?}", why);
                    }
                }
            }
        } else {
            error!("Bot user ID not found in TypeMap");
        }
    }
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Here's how the message handling works:

1. When a message is received, we first check if the bot is mentioned in the message using the `mentions_me` method.
2. If the bot is mentioned, we retrieve the bot's user ID from the `TypeMap` using the `BotUserId` key.
3. If the bot's user ID is found, we remove the mention from the message content to extract the actual query.
4. We pass the processed message content to the Rig agent's `process_message` method to generate a response.

- If the Rig agent successfully generates a response, we send it back to the channel where the message was received using `msg.channel_id.say`.
- If an error occurs during processing, we log the error and send an error message to the channel.
5. If the bot's user ID is not found in the `TypeMap`, we log an error.

This implementation allows users to interact with the bot by mentioning it in a message, providing a more natural way to ask questions and receive responses from the Rig agent.

#### Handling Bot Readiness

In the `ready` event handler, we set up slash commands and store the bot's user ID:  

```
async fn ready(&self, ctx: Context, ready: Ready) {
    info!("{} is connected!", ready.user.name);

    {
        let mut data = ctx.data.write().await;
        data.insert::<BotUserId>(ready.user.id);
    }

    let commands = Command::set_global_application_commands(&ctx.http, |commands| {
        commands
            .create_application_command(|command| {
                command
                    .name("hello")
                    .description("Say hello to the bot")
            })
            .create_application_command(|command| {
                command
                    .name("ask")
                    .description("Ask the bot a question")
                    .create_option(|option| {
                        option
                            .name("query")
                            .description("Your question for the bot")
                            .kind(CommandOptionType::String)
                            .required(true)
                    })
            })
    })
    .await;

    println!("Created the following global commands: {:#?}", commands);
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Here's what happens in the `ready` event handler:

1. When the bot successfully connects to Discord, the `ready` event is triggered.
2. We log a message indicating that the bot is connected, using the bot's name from the `Ready` struct.
3. We store the bot's user ID in the `TypeMap` using the `BotUserId` key. This allows us to access the bot's user ID in other event handlers.
4. We create global slash commands using the `Command::set_global_application_commands` method.

- We define two commands: `"hello"` and `"ask"`.
- The `"hello"` command is a simple command that greets the user.
- The `"ask"` command allows users to ask the bot a question. It has a required `"query"` option of type `String` where users can input their question.
5. We print the created global commands for debugging purposes.

### 5\. Implement the `main` Function

In the `main` function, we set up the bot and start it:  

```
#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let token = env::var("DISCORD_TOKEN").expect("Expected DISCORD_TOKEN in environment");

    let rig_agent = Arc::new(RigAgent::new().await?);

    let intents = GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::DIRECT_MESSAGES
        | GatewayIntents::MESSAGE_CONTENT;

    let mut client = Client::builder(&token, intents)
        .event_handler(Handler {
            rig_agent: Arc::clone(&rig_agent),
        })
        .await
        .expect("Err creating client");

    if let Err(why) = client.start().await {
        error!("Client error: {:?}", why);
    }

    Ok(())
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Here's a step-by-step breakdown of the main function:

1. We load environment variables from the `.env` file using `dotenv().ok()`.
2. We initialize the `tracing_subscriber` to set up logging with a maximum log level of `DEBUG`.
3. We retrieve the Discord bot token from the `DISCORD_TOKEN` environment variable.
4. We create a new instance of the `RigAgent` and wrap it in an `Arc` for thread-safe sharing.
5. We define the gateway intents, specifying the events we want to receive from Discord.
6. We create a new Discord client using the `Client::builder` method, passing in the bot token and intents.
7. We set the event handler for the client to an instance of the `Handler` struct, passing the `RigAgent` wrapped in an `Arc`.
8. We start the client using `client.start()` and handle any errors that may occur.

---

## Running and Testing the Bot

Now that our Discord bot is complete, let's run it and test its functionality.

### 1. Set Up Environment Variables

Create a `.env` file in the root of your project with the following content:  

```
DISCORD_TOKEN=your_discord_bot_token
OPENAI_API_KEY=your_openai_api_key
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Replace `your_discord_bot_token` with your actual Discord bot token and `your_openai_api_key` with your OpenAI API key.

> **Important**: Never commit your `.env` file or API keys to version control. Add `.env` to your `.gitignore` file to prevent accidental exposure.

### 2. Run the Bot

In your terminal, navigate to the project directory and run the following command:  

```
cargo run
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

If everything is set up correctly, you should see logs indicating that the bot is connected and the global commands have been created.

### 3. Invite the Bot to Your Discord Server

To invite the bot to your Discord server, follow these steps:

- Go to the [Discord Developer Portal](https://discord.com/developers/applications) and select your bot application.
- Navigate to the "OAuth2" section and click on "URL Generator".
- Under "Scopes", select `bot` and `applications.commands`.
- Under "Bot Permissions", select `Send Messages`, `Read Message History`, etc.
- Copy the generated URL and paste it into your browser.
- Select the server you want to invite the bot to and click "Authorize".

### 4. Test the Bot

Once the bot is running and invited to your server, you can test its functionality:

- **Slash Commands**:

- Type `/hello` to receive a greeting.
- Use `/ask` followed by a question to interact with the bot and receive a response generated by the Rig agent.
- **Mentions**:

- Mention the bot in a message with a question, like `@BotName How do I use Rig?`, and the bot will process your question and respond accordingly.

Here's two examples of the bot responding to questions:

[![AI Discord bot in action](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fvhvsg838ewy023a1e35t.png)](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fvhvsg838ewy023a1e35t.png)

## ![AI Discord bot in action](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Frm4tu6wi1es1wy19lbxt.png)

## Error Handling and Logging

Now that we've built and tested the bot, we need to ensure we are properly handling errors and logging our bot's behavior for improvement purposes. Rust provides powerful libraries like `anyhow` and `tracing` for error handling and logging.

### 1\. Error Handling with `anyhow`

The [`anyhow`](https://docs.rs/anyhow/) crate provides a flexible and easy-to-use error handling solution. It allows us to propagate and handle errors with additional context, making it easier to diagnose and fix issues. Here's an example of using `anyhow` in our `rig_agent.rs` file:  

```
use anyhow::{Context, Result};

// Example in rig_agent.rs
fn load_md_content<P: AsRef<Path>>(file_path: P) -> Result<String> {
    fs::read_to_string(file_path.as_ref())
        .with_context(|| format!("Failed to read markdown file: {:?}", file_path.as_ref()))
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

In this example, we use the `with_context` method to provide additional context to the error, specifying the file path that failed to be read. This context is included in the error message, making it easier to identify the source of the error.

### 2\. Logging with `tracing`

The [`tracing`](https://docs.rs/tracing/) crate provides a powerful and flexible logging solution that allows us to log messages at different levels of verbosity. Here's how we can set up logging in our `main.rs` file:  

```
use tracing::{info, error, debug};
use tracing_subscriber;

// Initialize tracing in main.rs
tracing_subscriber::fmt()
    .with_max_level(tracing::Level::DEBUG)
    .init();
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

In this example, we initialize the `tracing_subscriber` with a maximum log level of `DEBUG`. This means that all log messages with a severity level of `DEBUG` or higher will be captured and displayed.

Throughout our bot's code, we can use the `info!`, `error!`, and `debug!` macros to log messages at different severity levels, providing insights into the bot's behavior.

---

## Troubleshooting Common Issues

If you encounter errors, here are some common issues and how to resolve them:

- **API Key Errors**: Ensure your OpenAI API key and Discord token are correctly set in your `.env` file. Double-check for typos or extra spaces.
- **File Not Found**: If you receive a "Failed to read markdown file" error, check that your document paths are correct and the files exist in the `documents` directory.
- **Dependency Conflicts**: Run `cargo update` to ensure all dependencies are up to date.
- **Permission Errors**: Ensure your bot has the necessary permissions in your Discord server, such as sending messages and reading message history.

---

## Testing and Debugging Guidance
### Testing the Rig Agent Independently

Before integrating with Discord, you can test the Rig agent independently to ensure it's working as expected. Here's how:  

```
// Test the RigAgent in main.rs

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    let rig_agent = RigAgent::new().await?;
    let response = rig_agent.process_message("What is Rig?").await?;
    println!("Response: {}", response);

    Ok(())
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Run `cargo run` to see the output. This test confirms that the Rig agent can process messages and generate responses based on your knowledge base.

---

## Enhancing Your Bot

Congratulations! You've built a functional AI-powered Discord bot. Now, let's explore some ways to enhance it.

### 1. Expand the Knowledge Base

To make your bot more knowledgeable and versatile, consider adding more Markdown files to the `documents` directory. These files can cover a wide range of topics, from FAQs to technical documentation and beyond.

### 2. Customize the Bot's Behavior

The behavior of your bot is largely determined by the preamble defined in the `rig_agent.rs` file. By adjusting this preamble, you can fine-tune how the bot interacts with users, shaping its personality, tone, and overall approach to conversations. Experiment with different preambles to find the right balance for your bot's intended purpose and audience.

### 3. Add More Commands

Slash commands provide a structured and intuitive way for users to interact with your bot. Consider implementing additional commands in the `main.rs` file to extend your bot's functionality. For example, you could add commands for retrieving specific information, performing calculations, or triggering automated workflows.

### 4. Integrate Other APIs

To further enhance your bot's capabilities, consider integrating it with other APIs. For example, you could connect your bot to a weather API to provide real-time weather updates or integrate it with a news API to deliver the latest headlines. By leveraging external APIs, you can create powerful workflows and automate tasks within your Discord server.

> *Check out our guide on building agent tools and integrating APIs [here](https://rig.rs/build-with-rig-guide.html). You can also find more examples in the official [Rig repo](https://github.com/0xPlaygrounds/rig/tree/main/rig-core/examples).*

---

## Conclusion

In this guide, we've successfully built an AI-powered Discord bot using Rust and [Rig](https://rig.rs/). We've learned how to set up your environment, build a language model agent, integrate with Discord, and run your bot. With this foundation, you can continue to enhance and customize your bot, transforming it into a more robust system of autonomous agents to suit your needs.

See you in the next guide in the Build with Rig series!

---

## Further Resources

To deepen your understanding and continue building with Rig, check out these resources:

- [Rig Documentation](https://docs.rs/rig-core/latest/rig/)
- [Rig GitHub Repository](https://github.com/0xPlaygrounds/rig)
- [Examples Gallery Including This Walkthrough](https://github.com/0xPlaygrounds/awesome-rig/tree/main)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Understanding RAG Systems](https://www.promptingguide.ai/research/rag)

**Your Feedback Matters!** We're offering a unique opportunity to shape the future of Rig:

1. Build an AI-powered application using Rig.
2. Share your experience and insights via this [feedback form](https://bit.ly/Rig-Review).
3. Get a chance to win $100 and have your project featured in our showcase!

Your insights will directly influence Rig's growth. 🦀✨

---

## Code Files

For reference, here are the complete `rig_agent.rs` and `main.rs` files.

### `rig_agent.rs`

```
// rig_agent.rs

use anyhow::{Context, Result};
use rig::providers::openai;
use rig::vector_store::in_memory_store::InMemoryVectorStore;
use rig::vector_store::VectorStore;
use rig::embeddings::EmbeddingsBuilder;
use rig::agent::Agent;
use rig::completion::Prompt;
use std::path::Path;
use std::fs;
use std::sync::Arc;

pub struct RigAgent {
    agent: Arc<Agent<openai::CompletionModel>>,
}

impl RigAgent {
    pub async fn new() -> Result<Self> {
        // Initialize OpenAI client
        let openai_client = openai::Client::from_env();
        let embedding_model = openai_client.embedding_model(openai::TEXT_EMBEDDING_3_SMALL);

        // Create vector store
        let mut vector_store = InMemoryVectorStore::default();

        // Get the current directory and construct paths to markdown files
        let current_dir = std::env::current_dir()?;
        let documents_dir = current_dir.join("documents");

        let md1_path = documents_dir.join("Rig_guide.md");
        let md2_path = documents_dir.join("Rig_faq.md");
        let md3_path = documents_dir.join("Rig_examples.md");

        // Load markdown documents
        let md1_content = Self::load_md_content(&md1_path)?;
        let md2_content = Self::load_md_content(&md2_path)?;
        let md3_content = Self::load_md_content(&md3_path)?;

        // Create embeddings and add to vector store
        let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
            .simple_document("Rig_guide", &md1_content)
            .simple_document("Rig_faq", &md2_content)
            .simple_document("Rig_examples", &md3_content)
            .build()
            .await?;

        vector_store.add_documents(embeddings).await?;

        // Create index
        let index = vector_store.index(embedding_model);

        // Create Agent
        let agent = Arc::new(openai_client.agent(openai::GPT_4O)
            .preamble("You are an advanced AI assistant powered by Rig, a Rust library for building LLM applications. Your primary function is to provide accurate, helpful, and context-aware responses by leveraging both your general knowledge and specific information retrieved from a curated knowledge base.

                    Key responsibilities and behaviors:
                    1. Information Retrieval: You have access to a vast knowledge base. When answering questions, always consider the context provided by the retrieved information.
                    2. Clarity and Conciseness: Provide clear and concise answers. Ensure responses are short and concise. Use bullet points or numbered lists for complex information when appropriate.
                    3. Technical Proficiency: You have deep knowledge about Rig and its capabilities. When discussing Rig or answering related questions, provide detailed and technically accurate information.
                    4. Code Examples: When appropriate, provide Rust code examples to illustrate concepts, especially when discussing Rig's functionalities. Always format code examples for proper rendering in Discord by wrapping them in triple backticks and specifying the language as 'rust'. For example:
                        \`\`\`

rust
                        let example_code = \"This is how you format Rust code for Discord\";
                        println!(\"{}\", example_code);

                        \`\`\`
                    5. Keep your responses short and concise. If the user needs more information, they can ask follow-up questions.
                    ")
            .dynamic_context(2, index)
            .build());

        Ok(Self { agent })
    }

    fn load_md_content<P: AsRef<Path>>(file_path: P) -> Result<String> {
        fs::read_to_string(file_path.as_ref())
            .with_context(|| format!("Failed to read markdown file: {:?}", file_path.as_ref()))
    }

    pub async fn process_message(&self, message: &str) -> Result<String> {
        self.agent.prompt(message).await.map_err(anyhow::Error::from)
    }
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

### `main.rs`

```
// main.rs

mod rig_agent;

use anyhow::Result;
use serenity::async_trait;
use serenity::model::application::command::Command;
use serenity::model::application::interaction::{Interaction, InteractionResponseType};
use serenity::model::gateway::Ready;
use serenity::model::channel::Message;
use serenity::prelude::*;
use serenity::model::application::command::CommandOptionType;
use std::env;
use std::sync::Arc;
use tracing::{error, info, debug};
use rig_agent::RigAgent;
use dotenv::dotenv;

// Define a key for storing the bot's user ID in the TypeMap
struct BotUserId;

impl TypeMapKey for BotUserId {
    type Value = serenity::model::id::UserId;
}

struct Handler {
    rig_agent: Arc<RigAgent>,
}

#[async_trait]
impl EventHandler for Handler {
    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        debug!("Received an interaction");
        if let Interaction::ApplicationCommand(command) = interaction {
            debug!("Received command: {}", command.data.name);
            let content = match command.data.name.as_str() {
                "hello" => "Hello! I'm your helpful Rust and Rig-powered assistant. How can I assist you today?".to_string(),
                "ask" => {
                    let query = command
                        .data
                        .options
                        .get(0)
                        .and_then(|opt| opt.value.as_ref())
                        .and_then(|v| v.as_str())
                        .unwrap_or("What would you like to ask?");
                    debug!("Query: {}", query);
                    match self.rig_agent.process_message(query).await {
                        Ok(response) => response,
                        Err(e) => {
                            error!("Error processing request: {:?}", e);
                            format!("Error processing request: {:?}", e)
                        }
                    }
                }
                _ => "Not implemented :(".to_string(),
            };

            debug!("Sending response: {}", content);

            if let Err(why) = command
                .create_interaction_response(&ctx.http, |response| {
                    response
                        .kind(InteractionResponseType::ChannelMessageWithSource)
                        .interaction_response_data(|message| message.content(content))
                })
                .await
            {
                error!("Cannot respond to slash command: {}", why);
            } else {
                debug!("Response sent successfully");
            }
        }
    }

    async fn message(&self, ctx: Context, msg: Message) {
        if msg.mentions_me(&ctx.http).await.unwrap_or(false) {
            debug!("Bot mentioned in message: {}", msg.content);

            let bot_id = {
                let data = ctx.data.read().await;
                data.get::<BotUserId>().copied()
            };

            if let Some(bot_id) = bot_id {
                let mention = format!("<@{}>", bot_id);
                let content = msg.content.replace(&mention, "").trim().to_string();

                debug!("Processed content after removing mention: {}", content);

                match self.rig_agent.process_message(&content).await {
                    Ok(response) => {
                        if let Err(why) = msg.channel_id.say(&ctx.http, response).await {
                            error!("Error sending message: {:?}", why);
                        }
                    }
                    Err(e) => {
                        error!("Error processing message: {:?}", e);
                        if let Err(why) = msg
                            .channel_id
                            .say(&ctx.http, format!("Error processing message: {:?}", e))
                            .await
                        {
                            error!("Error sending error message: {:?}", why);
                        }
                    }
                }
            } else {
                error!("Bot user ID not found in TypeMap");
            }
        }
    }

    async fn ready(&self, ctx: Context, ready: Ready) {
        info!("{} is connected!", ready.user.name);

        {
            let mut data = ctx.data.write().await;
            data.insert::<BotUserId>(ready.user.id);
        }

        let commands = Command::set_global_application_commands(&ctx.http, |commands| {
            commands
                .create_application_command(|command| {
                    command
                        .name("hello")
                        .description("Say hello to the bot")
                })
                .create_application_command(|command| {
                    command
                        .name("ask")
                        .description("Ask the bot a question")
                        .create_option(|option| {
                            option
                                .name("query")
                                .description("Your question for the bot")
                                .kind(CommandOptionType::String)
                                .required(true)
                        })
                })
        })
        .await;

        println!("Created the following global commands: {:#?}", commands);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    let token = env::var("DISCORD_TOKEN").expect("Expected DISCORD_TOKEN in environment");

    let rig_agent = Arc::new(RigAgent::new().await?);

    let intents = GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::DIRECT_MESSAGES
        | GatewayIntents::MESSAGE_CONTENT;

    let mut client = Client::builder(&token, intents)
        .event_handler(Handler {
            rig_agent: Arc::clone(&rig_agent),
        })
        .await
        .expect("Err creating client");

    if let Err(why) = client.start().await {
        error!("Client error: {:?}", why);
    }

    Ok(())
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

------
title: "How to Deploy Your Rig App on AWS Lambda: A Step-by-Step Guide"
source: "https://dev.to/garance_buricatu_a6864136/how-to-deploy-your-rig-app-on-aws-lambda-a-step-by-step-guide-2ge5"
author:
  - "[[Garance Buricatu]]"
published: 2024-11-01
created: 2024-12-23
description: "TL;DR   A step-by-step walkthrough on deploying a simple AI Agent built with Rig, a fullstack agent... Tagged with rust, aws, llm, devops."
tags:
  - "clippings"
---
**TL;DR**

- A step-by-step walkthrough on deploying a simple AI Agent built with [Rig](https://github.com/0xPlaygrounds/rig), a fullstack agent framework, on AWS Lambda using the cargo lambda CLI.
- Comparison of performance metrics (memory usage, execution time, and cold starts) with a similar deployed Agent built with [LangChain](https://www.langchain.com/).
- **Stats: Rig Agent on AWS Lmabda :**
1. Low memory usage (26MB average)
2. Fast cold starts (90.9ms)
3. Consistent performance across memory configurations
- **Stats: LangChain Agent on AWS Lmabda:**
1. Higher memory usage (112-130MB)
2. Slower cold starts (1,898.52ms)
3. Performance improves with more memory allocation

## Table of Contents

- [How to Deploy Your Rig App on AWS Lambda: A Step-by-Step Guide](https://dev.to/garance_buricatu_a6864136/#how-to-deploy-your-rig-app-on-aws-lambda-a-step-by-step-guide)
- [Table of Contents](https://dev.to/garance_buricatu_a6864136/#table-of-contents)
- [Introduction](https://dev.to/garance_buricatu_a6864136/#introduction)
- [Prerequisites](https://dev.to/garance_buricatu_a6864136/#prerequisites)
- [AWS Lambda Quick Overview](https://dev.to/garance_buricatu_a6864136/#aws-lambda-quick-overview)
- [AWS and Rust](https://dev.to/garance_buricatu_a6864136/#aws-rust)
- [REST API backend](https://dev.to/garance_buricatu_a6864136/#rest-api-backend)
- [Event based task](https://dev.to/garance_buricatu_a6864136/#event-based-task)
- [Rig Entertainer Agent App](https://dev.to/garance_buricatu_a6864136/#rig-entertainer-agent-app)
- [Now let's deploy it!](https://dev.to/garance_buricatu_a6864136/#now-lets-deploy-it)
- [Metrics on the cloud](https://dev.to/garance_buricatu_a6864136/#metrics-on-the-cloud)
- [Deployment package](https://dev.to/garance_buricatu_a6864136/#deployment-package)
- [Memory, CPU, and runtime](https://dev.to/garance_buricatu_a6864136/#memory-cpu-and-runtime)
- [Cold starts](https://dev.to/garance_buricatu_a6864136/#cold-starts)
- [Langchain Entertainer Agent App](https://dev.to/garance_buricatu_a6864136/#langchain-entertainer-agent-app)
- [Deployment package](https://dev.to/garance_buricatu_a6864136/#deployment-package-1)
- [Memory, CPU, and runtime](https://dev.to/garance_buricatu_a6864136/#memory-cpu-and-runtime-1)
- [Cold starts](https://dev.to/garance_buricatu_a6864136/#cold-starts-1)
- [Community and Ecosystem](https://dev.to/garance_buricatu_a6864136/#community-and-ecosystem)
- [The Road Ahead: Rig's Future](https://dev.to/garance_buricatu_a6864136/#the-road-ahead-rigs-future)
- [Conclusion and Call to Action](https://dev.to/garance_buricatu_a6864136/#conclusion-and-call-to-action)

## Introduction

Welcome to the series **Deploy Your Rig Application**!  
Apps built with Rig can vary in complexity across three core dimensions: LLM usage, knowledge bases for RAG, and the compute infrastructure where the application is deployed. In this series, we’ll explore how different combinations of these dimensions can be configured for production use.

Today, we’ll start with a simple Rig agent that uses the [OpenAI model GPT-4-turbo](https://platform.openai.com/docs/models/gpt-4o), does not rely on a vector store (ie.: no RAGing), and will be deployed on AWS Lambda.

This blog will provide a step-by-step deployment guide for the simple Rig app, showcase performance metrics of the Rig app running on AWS Lambda, and compare these metrics with those of a [LangChain](https://dev.to/garance_buricatu_a6864136/\(https://www.langchain.com\)) app on the same platform.

> *💡 If you're new to Rig and want to start from the beginning or are looking for additional tutorials, check out our [blog series](https://rig.rs/build-with-rig-guide.html).*

Let’s dive in!

## Prerequisites

Before we begin building, ensure you have the following:

- A clone of the [`rig-entertainer-lambda`](https://github.com/garance-buricatu/rig-aws/tree/master/rig-entertainer-lambda) crate (or your own Rig application).
- An AWS account
- An Open AI api key

## AWS Lambda Quick Overview

You might deploy your Rust application on AWS lambda if it’s a task that can execute in under 15 mins or if your app is a REST API backend.

### AWS 🤝 Rust

AWS Lambda supports Rust through the use of the [OS-only runtime Amazon Linux 2023](https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtimes.html) (a lambda runtime) in conjunction with the [Rust runtime client](https://github.com/awslabs/aws-lambda-rust-runtime), a rust crate.

#### REST API backend

- Use the [`lambda-http`](https://github.com/awslabs/aws-lambda-rust-runtime/tree/main/lambda-http) crate (from the runtime client) to write your function’s entrypoint.
- Then, route traffic to your lambda via AWS API services like [Api Gateway](https://aws.amazon.com/api-gateway/), [App Sync](https://aws.amazon.com/pm/appsync), [VPC lattice](https://aws.amazon.com/vpc/lattice/), etc ...
- If your lambda handles multiple endpoints of your API, the crate [axum](https://github.com/tokio-rs/axum) facilitates the routing within the lambda.

#### Event based task (15 mins max.)

- Your lambda function is invoked by some event with the event passed as the payload. For example, configure your S3 bucket to trigger the lambda function when a new object is added to the bucket. The function will receive the new object in the payload and can further process it.
- Use the [`lambda_runtime`](https://github.com/awslabs/aws-lambda-rust-runtime/tree/main/lambda-runtime) crate with [`lambda_events`](https://github.com/awslabs/aws-lambda-rust-runtime/tree/main/lambda-events) (from the runtime client) to write your function’s entrypoint.
- Then, invoke your function either via [`lambda invoke` command](https://docs.aws.amazon.com/cli/latest/reference/lambda/invoke.html) or with integrated AWS triggers (ie. S3 UploadObject trigger).

> For both cases, the crate [`tokio`](https://docs.rs/tokio/latest/tokio/) must also be added to your project as the lambda runtime client uses `tokio` to handle asynchronous calls.

## Rig Entertainer Agent App 🤡

The crate [`rig-entertainer-lambda`](https://github.com/garance-buricatu/rig-aws-lambda/tree/master/rig-entertainer-lambda) implements a simple Rust program that is executed via the `lambda_runtime`. It invokes a `Rig` agent using the OpenAI API, to entertain users with jokes. It is an event-based task that I will execute with the `lambda invoke` command.

The main takeaway here is that the app's `Cargo.toml` file must include the following dependencies:

1. `rig-core` (our rig crate)
2. `lambda_runtime`
3. `tokio`

### Now let's deploy it!

There are *many* ways to deploy Rust lambdas to AWS. Some out of the box options include the AWS CLI, the [cargo lambda](https://www.cargo-lambda.info/guide/getting-started.html) CLI, the AWS SAM CLI, the AWS CDK, and more. You can also decide to create a Dockerfile for your app and use that container image in your Lambda function instead. See some useful examples [here](https://docs.aws.amazon.com/lambda/latest/dg/rust-package.html).

In this blog, we'll use the cargo lambda CLI option to deploy the code in `rig-entertainer-rust` from your local machine to an AWS lambda:  

```
# Add your AWS credentials to my terminal
# Create an AWS Lambda function named ‘rig-entertainer’ with architecture x86_64.

function_name='rig-entertainer'

cd rig-entertainer-lambda
cargo lambda build --release # Can define different architectures here with --arm64 for example
cargo lambda deploy $function_name # Since the name of the crate is the same as the the lambda function name, no need to specify a binary file
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

### Metrics on the cloud ☁️
#### Deployment package

This is the code configuration of the `rig-entertainer` function in AWS. The function’s code package (bundled code and dependencies required for lambda to run) includes the single rust binary called `bootstrap`, which is 3.2 MB.

[![Deployment Package Rust](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fjh72nanr3um3c8mpb8fh.png)](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fjh72nanr3um3c8mpb8fh.png)

#### Memory, CPU, and runtime

The image below gives metrics on memory usage and execution time of the function. Each row represents a single execution of the function. In **yellow** is the **total memory used**, in **red** is the amount of **memory allocated**, and in **blue** is the **runtime**.  
Although the lambda has many configuration options for the memory ranging from 128MB to 1024MB, we can see that the average memory used by our app is **26MB**.  
[![Rig Cloudwatch logs](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fqonxpllzxwtyfhgdrmy0.png)](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fqonxpllzxwtyfhgdrmy0.png)

Let's get more information on the metrics above by spamming the function and calculating averages. I invoked `rig-entertainer` 50 times for each memory configuration of 128MB, 256MB, 512MB, 1024MB using the [power tuner tool](https://github.com/alexcasalboni/aws-lambda-power-tuning) and the result of those invocations are displayed in the chart below.

The x-axis is the memory allocation, and the y-axis is the average runtime over the 50 executions of `rig-entertainer`.

> **Q.** We know that the function uses on average only 26MB per execution (which is less than the minimum memory allocation of 128MB) so why should we test higher memory configurations?
> 
> **A.** [vCPUs are added to the lambda in proportion to memory](https://docs.aws.amazon.com/lambda/latest/operatorguide/computing-power.html) so adding memory could still affect the performance.

However, we can see that adding memory to the function (and therefore adding computational power) does not affect its performance at all. Since the [cost of a lambda execution](https://aws.amazon.com/lambda/pricing/) is calculated in GB-seconds, we get the most efficient lambda for the lowest price!

[![Power Tuner Rust](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Ftr352m59k0vk7manl4jx.png)](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Ftr352m59k0vk7manl4jx.png)

#### Cold starts ❄️

[Cold starts](https://docs.aws.amazon.com/lambda/latest/operatorguide/execution-environments.html) occur when the lambda function's execution environment needs to be booted up from scratch. This includes setting up the actual compute that the lambda function is running on, and downloading the lambda function code and dependencies in that environment.

Cold start latency doesn't affect all function executions because once the lambda environment has been setup, it will be reused by subsequent executions of the same lambda.

In the lambda cloudwatch logs, if a function execution requires a cold start, we see the `Init Duration` metric at the end of the execution.

For `rig-entertainer`, we can see that the average cold start time is **90.9ms**:  
[![Rig cold starts](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fhcfgcrm0kipt5ruorior.png)](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fhcfgcrm0kipt5ruorior.png)

Note that the function was affected by cold starts 9 times out of the 245 times it was executed, so **0.036%** of the time.

## Langchain Entertainer Agent App 🐍

I replicated the OpenAI entertainer agent using the [langchain](https://python.langchain.com/) python library in this [mini python app](https://github.com/garance-buricatu/rig-aws-lambda/tree/master/langchain-entertainer-lambda) which I also deployed to AWS Lambda in a function called `langchain-entertainer`.

Let's compare the metrics outlined above.

#### Deployment package

This is the code configuration of the `langchain-entertainer` function in AWS. The function’s code package is a zip file including the lambda function code and all dependencies required for the lambda program to run.  
[![Deployment Package LangChain](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2F9x0w4arsxrvfdvjbkubm.png)](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2F9x0w4arsxrvfdvjbkubm.png)

#### Memory, CPU, and runtime

There are varying memory configurations from 128MB, 256MB, 512MB, to 1024MB for the lambda shown in the table below. When 128MB of memory is allocated, on average about **112MB** of memory is used, and when more more than 128MB is allocated, about **130MB** of memory is used and the **runtime is lower**.  
[![Cloudwatch Logs LangChain](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fv6f4m4skc2ndcz3913bo.png)](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fv6f4m4skc2ndcz3913bo.png)

Let's get some more averages for these metrics: I invoked `langchain-entertainer` 50 times for each memory configuration of 128MB, 256MB, 512MB, 1024MB using the [power tuner tool](https://github.com/alexcasalboni/aws-lambda-power-tuning) and the result of those invocations were plotted in the graph below.

We can see that by increasing the memory allocation (and therefore computation power) of `langchain-entertainer`, the function becomes more performant (lower runtime). However, note that since you pay per GB-seconds, a more performant function is more expensive.  
[![Power Tuner Langchain](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fm1mswjjhzfum2fn8v7dx.png)](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fm1mswjjhzfum2fn8v7dx.png)

#### Cold starts ❄️

For `langchain-entertainer`, the average cold start time is: **1,898.52ms**, ie. 20x as much as the rig app coldstart.  
[![Cold Starts LangChain](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fzysyfn6hyy3li0aw36bj.png)](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fzysyfn6hyy3li0aw36bj.png)

Note that the function was affected by cold starts 6 times out of the 202 times it was executed, so **0.029%** of the time.

## Resources

Rig is an emerging project in the open-source community, and we're continuously expanding its ecosystem with new integrations and tools. We believe in the power of community-driven development and welcome contributions from developers of all skill levels.

Stay connected and contribute to Rig's growth:

- 📚 [Documentation](https://docs.rs/rig-core/latest/rig/): Comprehensive guides and API references
- 💻 [GitHub Repository](https://github.com/0xPlaygrounds/rig): Contribute, report issues, or star the project
- 🌐 [Official Website](https://rig.rs/): Latest news, tutorials, and resources

Join our [community](https://discord.com/invite/playgrounds) channel to discuss ideas, seek help, and collaborate with other Rig developers.

## The Road Ahead: Rig's Future

As we continue to develop Rig, we're excited about the possibilities. Our roadmap includes:

1. **Expanding LLM Provider Support**: Adding integrations for more LLM providers to give developers even more choices.
2. **Enhanced Performance Optimizations**: Continuously improving Rig's performance to handle larger-scale applications.
3. **Advanced AI Workflow Templates**: Providing pre-built templates for common AI workflows to accelerate development further.
4. **Ecosystem Growth**: Developing additional tools and libraries that complement Rig's core functionality.

We're committed to making Rig the go-to library for LLM application development in Rust, and your feedback is crucial in shaping this journey.

## Conclusion and Call to Action

Rig is transforming LLM-powered application development in Rust by providing:

- A unified, intuitive API for multiple LLM providers
- High-level abstractions for complex AI workflows
- Type-safe development leveraging Rust's powerful features
- Extensibility and seamless ecosystem integration

We believe Rig has the potential to significantly enhance developers' building of AI applications, and we want you to be part of this journey.

**Your Feedback Matters!** We're offering a unique opportunity to shape the future of Rig:

1. Build an AI-powered application using Rig.
2. Share your experience and insights via this [feedback form](https://bit.ly/Rig-Review).
3. Get a chance to win $100 and have your project featured in our showcase!

Your insights will directly influence Rig's development, helping us create a tool that truly meets the needs of AI developers. 🦀✨---
title: "Introducing the New Wave Terminal v0.8"
source: "https://waveterm.hashnode.dev/introducing-the-new-wave-terminal-v08"
author:
  - "[[Mike Sawka]]"
published: 2024-09-25
created: 2024-12-23
description: "A modern, open-source terminal with file previews, an editor, AI, and web integration — designed to reduce context switching and keep you focused"
tags:
  - "clippings"
---
We built Wave to make you holistically more productive. With an all-new layout engine and integrated widgets, Wave seamlessly blends the command line with file browsing, editing, web integration, and AI into one powerful environment. By eliminating distractions and keeping everything at your fingertips, Wave helps you maintain a smooth, uninterrupted workflow. Whether you're previewing files, automating tasks, or leveraging AI, it's designed to streamline your process, allowing you to stay focused and reduce the need for constant context switching.

### PermalinkA Command Line Reimagined

Traditional command-line interfaces (CLIs) are powerful, but they often require switching between different applications for tasks like previewing files, browsing the web, or using AI tools. Wave brings all these functions together in one place with a flexible layout and keyboard shortcuts. You can preview remote files, pull up documentation, or browse the web without ever leaving your terminal. With integrated widgets and seamless navigation, Wave keeps you in the flow so you can get more done.

### PermalinkFaster, More Compatible, and Windows Native

Wave v0.8 is faster and more flexible than ever. It supports your standard terminal prompts and command completions while working smoothly with all shells (not just bash/zsh), removing the friction some users experienced with ohmyzsh packages. For the first time, Wave also runs **natively on Windows**, giving you a consistent experience across all major operating systems.

### PermalinkThis is Just the Beginning

Wave is open source and built with the community in mind. We're expanding its capabilities, adding support for creating your own widgets, and rolling out even more productivity features. Expect deeper AI integrations, better connections with external systems, and more ways to streamline your workflow. We're excited to continue evolving Wave, with your input helping to shape its future.

If you’re interested, please give us a star at [github.com/wavetermdev/waveterm](https://github.com/wavetermdev/waveterm) . You can download Wave at [waveterm.dev](https://waveterm.dev/) (available for MacOS, Linux, and Windows).---
title: "List of all items in this crate"
source: "https://docs.rs/rig-core/latest/rig/all.html"
author:
published:
created: 2024-12-23
description: "List of all items in this crate"
tags:
  - "clippings"
---
### Crate Items

- [Macros](https://docs.rs/rig-core/latest/rig/#macros "Macros")
- [Structs](https://docs.rs/rig-core/latest/rig/#structs "Structs")
- [Enums](https://docs.rs/rig-core/latest/rig/#enums "Enums")
- [Constants](https://docs.rs/rig-core/latest/rig/#constants "Constants")
- [Traits](https://docs.rs/rig-core/latest/rig/#traits "Traits")
- [Functions](https://docs.rs/rig-core/latest/rig/#functions "Functions")
- [Type Aliases](https://docs.rs/rig-core/latest/rig/#types "Type Aliases")# Rig
Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.

More information about this crate can be found in the [crate documentation](https://docs.rs/rig-core/latest/rig/).
## Table of contents

- [Rig](#rig)
  - [Table of contents](#table-of-contents)
  - [High-level features](#high-level-features)
  - [Installation](#installation)
  - [Simple example:](#simple-example)
  - [Integrations](#integrations)

## High-level features
- Full support for LLM completion and embedding workflows
- Simple but powerful common abstractions over LLM providers (e.g. OpenAI, Cohere) and vector stores (e.g. MongoDB, in-memory)
- Integrate LLMs in your app with minimal boilerplate

## Installation
```bash
cargo add rig-core
```

## Simple example:
```rust
use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and model
    // This requires the `OPENAI_API_KEY` environment variable to be set.
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.model("gpt-4").build();

    // Prompt the model and print its response
    let response = gpt4
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
```
Note using `#[tokio::main]` requires you enable tokio's `macros` and `rt-multi-thread` features
or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

## Integrations
Rig supports the following LLM providers natively:
- OpenAI
- Cohere
- Google Gemini
- xAI

Additionally, Rig currently has the following integration sub-libraries:
- MongoDB vector store: `rig-mongodb`
---
title: "Rig - Build Powerful LLM Applications in Rust"
source: "https://rig.rs/index.html#demo"
author:
  - "[[Playgrounds Analytics Inc.]]"
published:
created: 2024-12-23
description: "Rig: Build modular and scalable LLM Applications in Rust. Unified LLM interface, Rust-powered performance, and advanced AI workflow abstractions for efficient development."
tags:
  - "clippings"
---
[Skip to main content](https://rig.rs/#main)

[![Rig Logo - AI Development Framework](https://rig.rs/assets/PG-Logo.png)](https://rig.rs/#)

Build modular and scalable LLM Applications in Rust

`cargo add rig-core`

## Rig Demo

## Core Features of Rig

Consistent API across different LLM providers, simplifying integration and reducing vendor lock-in.

Leverage Rust's zero-cost abstractions and memory safety for high-performance LLM operations.

Implement complex AI systems like RAG and multi-agent setups with pre-built, modular components.

Utilize Rust's strong type system to ensure compile-time correctness in LLM interactions.

Built-in support for vector stores, enabling efficient similarity search and retrieval for AI applications.

Easy-to-use APIs for working with embeddings, crucial for semantic search and content-based recommendations.

## Why Developers Choose Rig for AI Development

### Efficient Development

- Type-safe API reduces runtime errors
- Async-first design for optimal resource utilization
- Seamless integration with Rust's ecosystem (Tokio, Serde, etc.)

### Production-Ready Architecture

- Modular design for easy customization and extension
- Comprehensive error handling with custom error types
- Built-in support for tracing and logging

## Connect with the Rig Community

[Discord](https://discord.com/invite/playgrounds) [Blog](https://substack.com/@tachi0x) [Twitter](https://x.com/Playgrounds0x) [GitHub](https://github.com/0xPlaygrounds/rig)---
title: "Rig - Build Powerful LLM Applications in Rust"
source: "https://rig.rs/index.html#demo"
author:
  - "[[Playgrounds Analytics Inc.]]"
published:
created: 2024-12-23
description: "Rig: Build modular and scalable LLM Applications in Rust. Unified LLM interface, Rust-powered performance, and advanced AI workflow abstractions for efficient development."
tags:
  - "clippings"
---
## Rig Demo

[See More](https://github.com/0xPlaygrounds/awesome-rig)

```rust
use rig::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct MovieReview {
    sentiment: String,
    rating: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = openai::Client::from_env();
    let extractor = client.extractor::<MovieReview>("gpt-4").build();
    let review = extractor.extract("I loved this movie! It's a solid 9/10.").await?;
    println!("Extracted: {:?}", review);
    Ok(())
}
```

## Core Features of Rig

### Unified LLM Interface

Consistent API across different LLM providers, simplifying integration and reducing vendor lock-in.

### Rust-Powered Performance

Leverage Rust's zero-cost abstractions and memory safety for high-performance LLM operations.

### Advanced AI Workflow Abstractions

Implement complex AI systems like RAG and multi-agent setups with pre-built, modular components.

### Type-Safe LLM Interactions

Utilize Rust's strong type system to ensure compile-time correctness in LLM interactions.

### Seamless Vector Store Integration

Built-in support for vector stores, enabling efficient similarity search and retrieval for AI applications.

### Flexible Embedding Support

Easy-to-use APIs for working with embeddings, crucial for semantic search and content-based recommendations.

## Why Developers Choose Rig for AI Development

### Efficient Development

- Type-safe API reduces runtime errors
- Async-first design for optimal resource utilization
- Seamless integration with Rust's ecosystem (Tokio, Serde, etc.)

### Production-Ready Architecture

- Modular design for easy customization and extension
- Comprehensive error handling with custom error types
- Built-in support for tracing and logging

## Connect with the Rig Community

[Discord](https://discord.com/invite/playgrounds) [Blog](https://substack.com/@tachi0x) [Twitter](https://x.com/Playgrounds0x) [GitHub](https://github.com/0xPlaygrounds/rig)---
title: "Rig - Build Powerful LLM Applications in Rust"
source: "https://rig.rs/index.html#demo"
author:
  - "[[Playgrounds Analytics Inc.]]"
published:
created: 2024-12-23
description: "Rig: Build modular and scalable LLM Applications in Rust. Unified LLM interface, Rust-powered performance, and advanced AI workflow abstractions for efficient development."
tags:
  - "clippings"
---
[Skip to main content](https://rig.rs/#main)

[![Rig Logo - AI Development Framework](https://rig.rs/assets/PG-Logo.png)](https://rig.rs/#)

Build modular and scalable LLM Applications in Rust

`cargo add rig-core`

## Rig Demo

## Core Features of Rig

Consistent API across different LLM providers, simplifying integration and reducing vendor lock-in.

Leverage Rust's zero-cost abstractions and memory safety for high-performance LLM operations.

Implement complex AI systems like RAG and multi-agent setups with pre-built, modular components.

Utilize Rust's strong type system to ensure compile-time correctness in LLM interactions.

Built-in support for vector stores, enabling efficient similarity search and retrieval for AI applications.

Easy-to-use APIs for working with embeddings, crucial for semantic search and content-based recommendations.

## Why Developers Choose Rig for AI Development

### Efficient Development

- Type-safe API reduces runtime errors
- Async-first design for optimal resource utilization
- Seamless integration with Rust's ecosystem (Tokio, Serde, etc.)

### Production-Ready Architecture

- Modular design for easy customization and extension
- Comprehensive error handling with custom error types
- Built-in support for tracing and logging

## Connect with the Rig Community

[Discord](https://discord.com/invite/playgrounds) [Blog](https://substack.com/@tachi0x) [Twitter](https://x.com/Playgrounds0x) [GitHub](https://github.com/0xPlaygrounds/rig)---
title: "Rig: A Rust Library for Building LLM-Powered Applications"
source: "https://dev.to/0thtachi/rig-a-rust-library-for-building-llm-powered-applications-3g75"
author:
  - "[[Tachi 0x]]"
published: 2024-09-01
created: 2024-12-23
description: "TL;DR    Rig is an open-source Rust library that simplifies and accelerates the development of... Tagged with rust, ai, opensource, llm."
tags:
  - "clippings"
---
**TL;DR**

- **Rig** is an open-source Rust library that simplifies and accelerates the development of powerful AI applications using Large Language Models (LLMs).
- **Key Features**: Unified API across LLM providers, advanced AI workflow support, flexible abstractions, and seamless integration with Rust's ecosystem.
- **Developer-Friendly**: Intuitive API design, comprehensive documentation, and scalability from simple chatbots to complex AI systems.
- **Join the Community**: Share your feedback and get a chance to win $100!

## Table of Contents

- [Introduction](https://dev.to/0thtachi/#introduction)
- [The Challenge of Building LLM Applications](https://dev.to/0thtachi/#the-challenge-of-building-llm-applications)
- [Enter Rig: A Game-Changer for LLM Development](https://dev.to/0thtachi/#enter-rig-a-game-changer-for-llm-development)
- [Getting Started with Rig](https://dev.to/0thtachi/#getting-started-with-rig)
- [Key Features and Developer Experience](https://dev.to/0thtachi/#key-features-and-developer-experience)
- [1\. Unified and Intuitive API](https://dev.to/0thtachi/#1-unified-and-intuitive-api)
- [2\. Advanced Abstractions for Complex Workflows](https://dev.to/0thtachi/#2-advanced-abstractions-for-complex-workflows)
- [3\. Type-Safe Development](https://dev.to/0thtachi/#3-type-safe-development)
- [4\. Extensibility and Integration](https://dev.to/0thtachi/#4-extensibility-and-integration)
- [Advanced Features: RAG Systems and Beyond](https://dev.to/0thtachi/#advanced-features-rag-systems-and-beyond)
- [Community and Ecosystem](https://dev.to/0thtachi/#community-and-ecosystem)
- [The Road Ahead: Rig's Future](https://dev.to/0thtachi/#the-road-ahead-rigs-future)
- [Conclusion and Call to Action](https://dev.to/0thtachi/#conclusion-and-call-to-action)

## Introduction

In the rapidly evolving landscape of artificial intelligence (AI), Large Language Models (LLMs) have emerged as powerful tools for building sophisticated AI applications. However, harnessing the full potential of LLMs often requires navigating complex APIs, managing different providers, and implementing intricate workflows. This is where Rig comes in – a comprehensive Rust library designed to transform how developers build LLM-powered applications.

## The Challenge of Building LLM Applications

Before diving into Rig's capabilities, let's consider the challenges developers face when building LLM applications:

1. **API Complexity**: Each LLM provider has its own API, requiring developers to learn and manage multiple interfaces.
2. **Workflow Management**: Implementing advanced AI workflows, such as Retrieval-Augmented Generation (RAG), involves multiple steps and can be error-prone.
3. **Performance and Scalability**: Ensuring optimal performance and scalability in LLM applications can be challenging, especially as projects grow in complexity.
4. **Type Safety and Error Handling**: Maintaining type safety and robust error handling across different LLM interactions is crucial but often difficult.

## Enter Rig: A Game-Changer for LLM Application Development

Rig is more than just an API wrapper; it's a comprehensive framework that addresses these challenges head-on. By providing high-level abstractions and a unified interface, Rig simplifies the development process, allowing you to focus on building innovative AI solutions rather than wrestling with implementation details.

Whether you're a seasoned Rust developer or new to the language, Rig offers a range of features designed to make your LLM application development smoother, faster, and more enjoyable.

## Getting Started with Rig

Let's dive into a simple example to demonstrate how easy it is to get started with Rig:  

```
use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize the OpenAI client using environment variables
    let openai_client = openai::Client::from_env();

    // Create a GPT-4 model instance
    let gpt4 = openai_client.model("gpt-4").build();

    // Send a prompt to GPT-4 and await the response
    let response = gpt4.prompt("Explain quantum computing in one sentence.").await?;

    // Print the response
    println!("GPT-4: {}", response);

    Ok(())
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This simple example demonstrates how Rig abstracts away the complexities of interacting with OpenAI's API, allowing you to focus on the core logic of your application.

To include Rig in your project, add the following to your `Cargo.toml`:  

```
[dependencies]
rig-core = "0.0.6"
tokio = { version = "1.34.0", features = ["full"] }
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

> 💡 **Tip**: Don't forget to set the `OPENAI_API_KEY` environment variable before running your application.

## Key Features and Developer Experience

Rig combines Rust's powerful type system and performance with intuitive abstractions tailored for AI development. Let's explore some of its key features:

### 1. Unified and Intuitive API

One of Rig's standout features is its consistent interface across different LLM providers:  

```
// Using OpenAI
let gpt4 = openai_client.model("gpt-4").build();
let response = gpt4.prompt("Hello, GPT-4!").await?;

// Using Cohere
let command = cohere_client.model("command").build();
let response = command.prompt("Hello, Cohere!").await?;
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This unified API design ensures that switching between providers or adding new ones to your project is seamless, reducing cognitive load and improving code maintainability.

### 2. Advanced Abstractions for Complex Workflows

Rig shines when it comes to implementing complex AI workflows. For example, creating a Retrieval-Augmented Generation (RAG) system typically involves multiple steps:

1. Generating embeddings for documents
2. Storing these embeddings in a vector database
3. Retrieving relevant context based on user queries
4. Augmenting the LLM prompt with this context

With Rig, this entire process can be condensed into a few lines of code:  

```
let rag_agent = openai_client.context_rag_agent("gpt-4")
    .preamble("You are a helpful assistant.")
    .dynamic_context(2, vector_store.index(embedding_model))
    .build();

let response = rag_agent.prompt("What is the capital of France?").await?;
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This high-level abstraction allows developers to implement advanced AI systems quickly and efficiently, without getting bogged down in the implementation details.

### 3. Type-Safe Development

Leveraging Rust's strong type system, Rig provides compile-time guarantees and better auto-completion, enhancing the developer experience:  

```
#[derive(serde::Deserialize, JsonSchema)]
struct Person {
    name: String,
    age: u8,
}

let extractor = openai_client.extractor::<Person>("gpt-4").build();
let person: Person = extractor.extract("John Doe is 30 years old").await?;
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This type-safe approach helps catch errors early in the development process and makes refactoring and maintenance easier.

### 4. Extensibility and Integration

Rig's flexible architecture allows for easy customization and seamless integration with Rust's growing AI ecosystem:  

```
impl VectorStore for MyCustomStore {
    // Implementation details...
}

let my_store = MyCustomStore::new();
let rag_agent = openai_client.context_rag_agent("gpt-4")
    .dynamic_context(2, my_store.index(embedding_model))
    .build();
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This extensibility ensures that Rig can grow with your project's needs and integrate with other tools in your AI development stack.

## Advanced Features: RAG Systems and Beyond

Let's explore a more comprehensive example of a RAG system with Rig, showcasing its ability to handle complex AI workflows:  

```
use rig::{
    completion::Prompt,
    embeddings::EmbeddingsBuilder,
    providers::openai::Client,
    vector_store::{in_memory_store::InMemoryVectorStore, VectorStore},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize OpenAI client and embedding model
    let openai_client = Client::from_env();
    let embedding_model = openai_client.embedding_model("text-embedding-ada-002");

    // Create and populate vector store
    let mut vector_store = InMemoryVectorStore::default();
    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .simple_document("doc1", "Rig is a Rust library for building LLM applications.")
        .simple_document("doc2", "Rig supports OpenAI and Cohere as LLM providers.")
        .build()
        .await?;
    vector_store.add_documents(embeddings).await?;

    // Create and use RAG agent
    let rag_agent = openai_client.context_rag_agent("gpt-4")
        .preamble("You are an assistant that answers questions about Rig.")
        .dynamic_context(1, vector_store.index(embedding_model))
        .build();

    let response = rag_agent.prompt("What is Rig?").await?;
    println!("RAG Agent: {}", response);

    Ok(())
}
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

This example demonstrates how Rig abstracts the complexity of creating a RAG system, handling embedding generation, vector storage, and context retrieval efficiently. With just a few lines of code, you've implemented a sophisticated AI system that can provide context-aware responses.

But Rig's capabilities extend beyond RAG systems. Its flexible architecture allows for the implementation of various AI workflows, including:

- Multi-agent systems for complex problem-solving
- AI-powered data analysis and extraction
- Automated content generation and summarization
- And much more!

## Community and Ecosystem

Rig is an emerging project in the open-source community, and we're continuously expanding its ecosystem with new integrations and tools. We believe in the power of community-driven development and welcome contributions from developers of all skill levels.

Stay connected and contribute to Rig's growth:

- 📚 [Documentation](https://docs.rs/rig-core/latest/rig/): Comprehensive guides and API references
- 💻 [GitHub Repository](https://github.com/0xPlaygrounds/rig): Contribute, report issues, or star the project
- 🌐 [Official Website](https://rig.rs/): Latest news, tutorials, and resources

Join our [community](https://discord.com/invite/playgrounds) channel to discuss ideas, seek help, and collaborate with other Rig developers.

## The Road Ahead: Rig's Future

As we continue to develop Rig, we're excited about the possibilities. Our roadmap includes:

1. **Expanding LLM Provider Support**: Adding integrations for more LLM providers to give developers even more choices.
2. **Enhanced Performance Optimizations**: Continuously improving Rig's performance to handle larger-scale applications.
3. **Advanced AI Workflow Templates**: Providing pre-built templates for common AI workflows to accelerate development further.
4. **Ecosystem Growth**: Developing additional tools and libraries that complement Rig's core functionality.

We're committed to making Rig the go-to library for LLM application development in Rust, and your feedback is crucial in shaping this journey.

## Conclusion and Call to Action

Rig is transforming LLM-powered application development in Rust by providing:

- A unified, intuitive API for multiple LLM providers
- High-level abstractions for complex AI workflows
- Type-safe development leveraging Rust's powerful features
- Extensibility and seamless ecosystem integration

We believe Rig has the potential to significantly enhance developers' building of AI applications, and we want you to be part of this journey.

**Your Feedback Matters!** We're offering a unique opportunity to shape the future of Rig:

1. Build an AI-powered application using Rig.
2. Share your experience and insights via this [feedback form](https://bit.ly/Rig-Review).
3. Get a chance to win $100 and have your project featured in our showcase!

Your insights will directly influence Rig's development, helping us create a tool that truly meets the needs of AI developers. 🦀✨---
title: "Vector indexes - Cypher Manual"
source: "https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/"
author:
  - "[[Neo4j Graph Data Platform]]"
published:
created: 2024-12-23
description: "Information about creating, querying, and deleting vector indexes with Cypher."
tags:
  - "clippings"
---
Vector indexes allow you to query vector embeddings from large datasets. An embedding is a numerical representation of a data object, such as a text, image, or document. Each word or token in a text is typically represented as high-dimensional vector where each dimension represents a certain aspect of the word’s meaning.

The embedding for a particular data object can be created by both proprietary (such as [Vertex AI](https://cloud.google.com/vertex-ai) or [OpenAI](https://openai.com/)) and open source (such as [sentence-transformers](https://github.com/UKPLab/sentence-transformers)) embedding generators, which can produce vector embeddings with dimensions such as 256, 768, 1536, and 3072. In Neo4j, vector embeddings are stored as `LIST<INTEGER | FLOAT>` properties on a node or relationship.

For example, the movie The Godfather, has the following `plot`: `"The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son."` This is its 1536-dimensional `embedding` property, where each element in the `LIST` represents a particular aspect of the plot’s meaning:

```
[0.005967312026768923, -0.03817005082964897, 0.0014667075593024492, -0.03868866711854935, -0.006505374796688557, 0.020900176838040352, -0.0027551413513720036, -0.0024731445591896772, -0.03734026849269867, -0.02228747308254242, 0.028783122077584267, 0.017905177548527718, 0.011396560817956924, 0.014235977083444595, 0.023143187165260315, -0.014184115454554558, 0.029846282675862312, -0.011928141117095947, 0.018838683143258095, -0.0019172541797161102, 0.0033483069855719805, 0.009497134014964104, -0.03516208380460739, 0.0021441481076180935, 0.002657901030033827, 0.0030760341323912144, 0.004255882930010557, -0.020809419453144073, 0.02358401007950306, -0.013808120042085648, 0.01064456906169653, -0.006975369527935982, 0.007318951655179262, -0.013872946612536907, 0.005905726458877325, -0.010689947754144669, 0.0020225979387760162, -0.016245609149336815, -0.00038815077277831733, -0.007163367234170437, 0.027668101713061333, 0.007215228863060474, -0.009380445815622807, -0.02956104464828968, -0.000863007502630353, 0.012142069637775421, 0.0012957267463207245, -0.027953339740633965, -0.016414159908890724, 0.008453421294689178, -0.0010777463903650641, 0.03311355784535408, -0.013639570213854313, -0.052457891404628754, 0.0010242642601951957, 0.0034390646032989025, -0.01049546804279089, 0.006456754636019468, 0.003970644902437925, -0.011629937216639519, 0.005280147306621075, -0.023402493447065353, -0.014689764939248562, -0.007623638026416302, -0.002453696448355913, 0.02290981076657772, 0.0017989451298490167, 0.0013427261728793383, -0.001776255783624947, -0.002414800226688385, 0.04833490028977394, 0.031142819672822952, -0.0033013075590133667, 0.017879245802760124, 0.0070077828131616116, -0.016154851764440536, -0.005772831384092569, 0.019875913858413696, -0.018008900806307793, 0.012764407321810722, 0.0055232481099665165, -0.027901478111743927, -0.0034909259993582964, 0.0307279285043478, 0.006472961511462927, 0.008861830458045006, -0.01802186481654644, 0.018281172960996628, -0.014223011210560799, -0.00018313586770091206, 0.0026352116838097572, 0.0006754148053005338, 0.014975002966821194, 0.024361930787563324, -0.017166150733828545, 0.0028880364261567593, 0.011824417859315872, 0.01710132323205471, -0.0005003822734579444, -0.018890544772148132, -0.002192768268287182, -0.0018264965619891882, 0.011033530347049236, -0.009095207788050175, -0.022689398378133774, -0.004281813744455576, 0.007092057727277279, -0.015247276052832603, 0.024115590378642082, 0.002996621420606971, -0.02834230102598667, 0.030546413734555244, 0.02350621670484543, -0.020511215552687645, 0.010190781205892563, -0.016582708805799484, 0.028238577768206596, -0.011967036873102188, 0.011623455211520195, -0.02797926962375641, 0.0026254875119775534, 0.018307102844119072, 0.0038701631128787994, -0.03850715234875679, 0.006246067117899656, -0.0006312514888122678, 0.010352848097682, -0.02358401007950306, -0.026708664372563362, -0.002863726345822215, 0.035862214863300323, 0.009860164485871792, -0.01726987399160862, 0.004275330808013678, -0.02663087099790573, 0.009140586480498314, -0.013872946612536907, 0.019136887043714523, -0.020835351198911667, -0.0250879917293787, 0.03044269047677517, 0.026280807331204414, -0.013406192883849144, 0.006683648563921452, -0.01216800045222044, 0.007824601605534554, 0.031505849212408066, 0.023726629093289375, 0.0294832531362772, -0.013678465969860554, 0.033891480416059494, 0.009211895987391472, 0.017088359221816063, -0.02183368429541588, 0.01847565360367298, 0.004644844215363264, -0.009834233671426773, -0.011344699189066887, -0.0006725785788148642, 0.00012691882147919387, 0.015338033437728882, 0.025736261159181595, -0.003967403434216976, -0.007312469184398651, -0.01312743779271841, 0.02350621670484543, -0.0006843284936621785, -0.011785522103309631, 0.006570201832801104, -0.004187814891338348, -0.0070013003423810005, 0.0165178831666708, -0.004537879955023527, 0.022715330123901367, -0.0025120405480265617, 0.025580676272511482, 0.005053253378719091, -0.0020063910633325577, -0.039285074919462204, -0.001816772622987628, 0.0007224142318591475, 0.0161029901355505, 0.04086684808135033, 0.03536953032016754, 0.009626788087189198, -0.023571044206619263, -0.009607339277863503, 0.011085391975939274, 0.020835351198911667, -0.0009027139167301357, -0.007584741804748774, 0.016958704218268394, 0.011130770668387413, -0.016829051077365875, -0.6712950468063354, -0.006511857267469168, -0.024854615330696106, -0.02663087099790573, -0.00008933950448408723, 0.0061779990792274475, 0.004605947993695736, 0.013231161050498486, -0.020187081769108772, 0.00798666849732399, -0.001847565290518105, 0.04086684808135033, 0.007519915234297514, 0.0040808506309986115, -0.034021131694316864, -0.01997963711619377, -0.004972219467163086, -0.023220978677272797, 0.012129104696214199, 0.0018329792656004429, -0.011649386025965214, 0.028446022421121597, -0.0010356089333072305, -0.006223377771675587, 0.021211346611380577, 0.004006299655884504, 0.021937407553195953, -0.02927580662071705, -0.01129283756017685, -0.009296170435845852, -0.01864420250058174, 0.02717541716992855, -0.0003555347793735564, 0.0021700789220631123, 0.048360832035541534, -0.002277043182402849, -0.009049829095602036, 0.033969271928071976, 0.004557327833026648, 0.018916476517915726, -0.000779542897362262, -0.00638544512912631, 0.022183749824762344, -0.012757924385368824, -0.027149485424160957, -0.012278205715119839, 0.0238303504884243, -0.02963883802294731, 0.005218561738729477, -0.004434156697243452, 0.013665501028299332, -0.0024520757142454386, 0.002124700229614973, -0.007273572962731123, -0.0035654769744724035, -0.0028621056117117405, 0.020640870556235313, 0.01091684214770794, -0.0006867594784125686, -0.011694764718413353, 0.011215046048164368, 0.016504917293787003, 0.00827838946133852, -0.0044471221044659615, 0.010676982812583447, 0.027771824970841408, -0.0133802630007267, 0.029820352792739868, 0.008349698968231678, -0.014573076739907265, -0.009017415344715118, 0.011655868031084538, -0.0061066895723342896, -0.013082059100270271, 0.004353123251348734, 0.00672254478558898, 0.01773662678897381, 0.012433790601789951, 0.023843316361308098, 0.015221345238387585, -0.0046221548691391945, -0.00026214358513243496, -0.016582708805799484, 0.016504917293787003, 0.028005201369524002, 0.005516765173524618, -0.04309689253568649, 0.013743292540311813, -0.0064308238215744495, -0.007176332641392946, 0.01911095716059208, 0.00446332897990942, -0.012971853837370872, -0.016919808462262154, 0.010048162192106247, 0.0032769974786788225, -0.021548446267843246, 0.001816772622987628, 0.01856641098856926, -0.04804966226220131, 0.007286538369953632, -0.007299503777176142, -0.014080392196774483, 0.008952588774263859, 0.023908143863081932, 0.012932957150042057, -0.008433973416686058, 0.012783855199813843, 0.0430709607899189, -0.01015836838632822, 0.03534360229969025, -0.007584741804748774, -0.016453055664896965, -0.005720969755202532, -0.014871280640363693, -0.026540113613009453, 0.005228285677731037, 0.0004019264888484031, 0.005931657273322344, -0.02533433400094509, -0.018825719133019447, 0.0023353875149041414, 0.0014059323584660888, -0.02020004764199257, 0.022481953725218773, 0.034980569034814835, -0.02709762565791607, -0.022974636405706406, -0.025023166090250015, 0.00641785841435194, -0.00019822835747618228, -0.004845807328820229, 0.0003723492263816297, -0.010132437571883202, 0.01498796883970499, 0.001948046963661909, -0.0020161152351647615, -0.008842382580041885, 0.0223652645945549, -0.013574742712080479, -0.002369421534240246, 0.003275376744568348, 0.005879795644432306, 0.005789037793874741, 0.006359514314681292, -0.03549918532371521, 0.003118171589449048, -0.026993902400135994, -0.01614188589155674, 0.011578075587749481, 0.0008524731383658946, -0.013367297127842903, 0.004194297362118959, 0.019331367686390877, 0.006152068264782429, -0.015208380296826363, -0.0018005658639594913, -0.015714028850197792, -0.01681608520448208, -0.028990568593144417, 0.010676982812583447, 0.024595309048891068, -0.045560311526060104, -0.0009262136882171035, 0.014845349825918674, -0.020887212827801704, 0.015739960595965385, 0.011727177537977695, 0.0012560202740132809, -0.023052429780364037, 0.0014245701022446156, -0.013062611222267151, -0.011299320496618748, 0.022274507209658623, 0.011338216252624989, -0.007908876053988934, 0.010339883156120777, -0.006132620386779308, 0.01247916929423809, -0.007947771809995174, -0.0025347298942506313, -0.011416008695960045, 0.011027047410607338, 0.004521673079580069, 0.04880165681242943, 0.0012543996563181281, 0.02115948498249054, 0.0165178831666708, -0.025373229756951332, 0.026125222444534302, -0.0031262750271707773, 0.007669016718864441, 0.003821542952209711, -0.021561412140727043, 0.008187631145119667, 0.02358401007950306, 0.02249491773545742, 0.015247276052832603, -0.004560569301247597, 0.030753860250115395, 0.031090958043932915, -0.021457688882946968, 0.027694031596183777, -0.004823117982596159, 0.0049171168357133865, -0.018346000462770462, -0.0030355174094438553, -0.011176149360835552, 0.024102624505758286, 0.006923507899045944, 0.010009266436100006, -0.00510187353938818, 0.0007916979375295341, -0.004722636193037033, 0.019914809614419937, 0.026190048083662987, -0.013289504684507847, 0.006346548907458782, -0.015415825881063938, -0.026734594255685806, 0.003623821074143052, 0.005325525999069214, -0.003922024741768837, -0.00640813447535038, -0.014624938368797302, -0.0065021333284676075, 0.007435640320181847, -0.002808623481541872, 0.010138919577002525, -0.033813685178756714, -0.0032008260022848845, 0.01614188589155674, -0.018994268029928207, 0.008135770447552204, -0.008596041239798069, -0.015662167221307755, 0.004310985561460257, -0.014663834124803543, 0.014962038025259972, -0.03479905426502228, 0.013114472851157188, 0.01341915875673294, 0.05092797800898552, -0.011908693239092827, 0.005332008935511112, -0.013367297127842903, 0.02501020021736622, -0.00029678543796762824, -0.02454344742000103, 0.003152205841615796, -0.015454721637070179, 0.010028714314103127, -0.02102983184158802, -0.0032624113373458385, 0.03583628311753273, -0.015026864595711231, 0.00672254478558898, 0.000010907877367571928, 0.019875913858413696, 0.020161151885986328, 0.014054462313652039, -0.005675591062754393, -0.009224860928952694, 0.014793488197028637, 0.03687351569533348, -0.005442214198410511, 0.005633453372865915, -0.0030436208471655846, -0.012615305371582508, -0.009075759910047054, 0.017192082479596138, -0.002220319816842675, 0.005798762198537588, -0.0007568534929305315, 0.010378778912127018, 0.005908967927098274, -0.0158825796097517, 0.0088812792673707, 0.007766257040202618, -0.0030209312681108713, -0.013561777770519257, -0.035395462065935135, 0.022391194477677345, -0.0027049004565924406, 0.004748567007482052, -0.020433424040675163, -0.00028706141165457666, -0.005092149134725332, -0.018371930345892906, 0.006009449250996113, -0.00645027169957757, 0.015286171808838844, -0.012343033216893673, -0.008628454059362411, -0.010605673305690289, 0.009192448109388351, 0.007500466890633106, -0.013535846956074238, 0.003831267124041915, -0.02956104464828968, 0.0009724028059281409, 0.0034585127141326666, -0.00004074468961334787, -0.025139853358268738, 0.012278205715119839, 0.023519182577729225, -0.012913509272038937, -0.006301170215010643, 0.0037178201600909233, 0.004716153722256422, -0.017905177548527718, 0.009769407100975513, -0.019746258854866028, -0.011675315909087658, 0.007409709505736828, -0.022676432505249977, -0.013406192883849144, 0.003922024741768837, 0.03925914317369461, -0.011325251311063766, -0.014611972495913506, -0.022404160350561142, -0.03311355784535408, 0.0024634203873574734, 0.1057974249124527, 0.014145219698548317, 0.025956671684980392, 0.006878129206597805, -0.019914809614419937, -0.019162818789482117, -0.009231343865394592, -0.04423784464597702, 0.012018898501992226, -0.00921837892383337, 0.02408965863287449, -0.026501217857003212, 0.020225977525115013, 0.005014357157051563, 0.02053714729845524, 0.014521215111017227, -0.002670866437256336, -0.020433424040675163, -0.0015372068155556917, -0.031168751418590546, 0.0051213214173913, 0.006865163799375296, 0.010048162192106247, 0.003795612370595336, -0.009749959222972393, -0.024063728749752045, 0.026449356228113174, 0.00967864878475666, -0.009049829095602036, -0.012284688651561737, -0.02475089207291603, 0.0034844432957470417, -0.00928320549428463, 0.011772556230425835, -0.01811262220144272, -0.01918874867260456, 0.009043346159160137, 0.023843316361308098, 0.02580108679831028, 0.005980277433991432, 0.029327668249607086, -0.008103356696665287, 0.008083908818662167, -0.005490834359079599, 0.021146519109606743, -0.0023499734234064817, -0.03298390284180641, 0.005283388774842024, -0.00043352958164177835, -0.024271173402667046, 0.03181701898574829, -0.000028944177756784484, -0.004479535389691591, -0.002066355897113681, 0.017995934933423996, -0.012783855199813843, 0.013859981670975685, -0.006615580525249243, -0.0008403180981986225, 0.025489918887615204, -0.01789221167564392, -0.03189481049776077, 0.00028949242550879717, -0.03251715004444122, 0.03588814660906792, -0.03500650078058243, -0.007869980297982693, -0.024361930787563324, 0.00451519014313817, -0.018177449703216553, 0.020627904683351517, 0.003249445930123329, 0.010962220840156078, -0.005299595184624195, 0.048023734241724014, -0.0033094107639044523, 0.012971853837370872, -0.02290981076657772, 0.017918141558766365, -0.016245609149336815, -0.013179299421608448, -0.020589008927345276, 0.0037469922099262476, -0.029327668249607086, -0.007383778691291809, 0.013017232529819012, 0.006327101029455662, -0.02689017914235592, -0.004385536536574364, 0.005789037793874741, -0.005597798619419336, -0.004152160137891769, 0.012719028629362583, -0.008220044896006584, -0.01702353172004223, -0.011506766080856323, 0.0042980206198990345, 0.0018702547531574965, -0.0032964455895125866, 0.007267090491950512, -0.009581409394741058, -0.0058182100765407085, -0.005429248791188002, -0.008829417638480663, -0.0030403793789446354, -0.01194110605865717, -0.002591453492641449, 0.005756624508649111, -0.01618078351020813, -0.009821268729865551, -0.00021210535487625748, -0.01768476516008377, -0.0005562954465858638, -0.017451388761401176, -0.015545479021966457, 0.02332470193505287, 0.015960371121764183, 0.02208002656698227, 0.01369143184274435, -0.014495284296572208, -0.007701430004090071, -0.0005567006301134825, 0.027590308338403702, 0.05188741534948349, -0.023609939962625504, -0.017957039177417755, 0.015999266877770424, -0.020900176838040352, 0.003038758644834161, 0.021042795851826668, -0.009814785793423653, 0.0014083633432164788, 0.010897394269704819, -0.0167253278195858, -0.020135220140218735, -0.005273664370179176, -0.009788854978978634, -0.002986897248774767, -0.008764590136706829, -0.006729027256369591, -0.018449721857905388, -0.009166517294943333, -0.002651418326422572, 0.008245975710451603, 0.0034358231350779533, -0.028757192194461823, 0.01511762198060751, -0.008544179610908031, 0.005344973877072334, 0.013924808241426945, -0.003299686824902892, -0.04143732413649559, -0.0008403180981986225, 0.010949255898594856, -0.013600673526525497, -0.03448788449168205, -0.007863497361540794, -0.01809965819120407, -0.00444063963368535, 0.004920358303934336, 0.0330357663333416, -0.008816451765596867, 0.006683648563921452, 0.00823301076889038, -0.015947405248880386, 0.02608632668852806, 0.0037243026308715343, -0.007623638026416302, -0.028031131252646446, 0.027123555541038513, 0.01843675784766674, 0.016712361946702003, 0.040374163538217545, -0.0021538722794502974, 0.01885164901614189, -0.011740143410861492, 0.017490284517407417, -0.0004517621418926865, -0.00034439266892150044, -0.026190048083662987, -0.021729961037635803, 0.0020209772046655416, -0.014521215111017227, -0.01467679999768734, 0.002505557844415307, -0.01061863824725151, 0.015623271465301514, -0.010087057948112488, -0.0031748951878398657, 0.01631043665111065, 0.016375262290239334, -0.013257091864943504, 0.010741809383034706, -0.012932957150042057, -0.002484489232301712, 0.0027324517723172903, 0.00897203665226698, -0.004793945699930191, 0.0043466403149068356, -0.0020047705620527267, 0.0021538722794502974, 0.021263208240270615, -0.0269679706543684, -0.024115590378642082, -0.0025833500549197197, 0.030598275363445282, 0.002772968728095293, 0.01584368385374546, 0.006981851998716593, -0.0037113374564796686, -0.01273199450224638, -0.026280807331204414, -0.02182071842253208, -0.049527715891599655, 0.02195037342607975, -0.008628454059362411, -0.004353123251348734, 0.01064456906169653, -0.009698097594082355, -0.04094463959336281, 0.0238303504884243, 0.0034649954177439213, 0.032802388072013855, 0.0002048123424174264, 0.022507883608341217, 0.03770329803228378, -0.010346366092562675, 0.0028588641434907913, 0.026410460472106934, 0.019085025414824486, 0.008848865516483784, 0.015830717980861664, -0.004469811450690031, 0.013808120042085648, -0.012031864374876022, -0.02099093608558178, -0.006054827943444252, -0.045638103038072586, -0.024050762876868248, 0.014417491853237152, 0.01218744833022356, 0.0032413427252322435, -0.013302470557391644, -0.0003156257444061339, 0.006942956242710352, 0.00542600778862834, -0.0034358231350779533, 0.022067060694098473, -0.013847015798091888, -0.026942040771245956, -0.0334506556391716, -0.01835896447300911, -0.0021036313846707344, -0.001962633104994893, 0.012615305371582508, -0.0186053067445755, 0.01572699472308159, -0.02542509138584137, 0.019422125071287155, -0.013950739055871964, -0.002110114088281989, 0.02052418142557144, -0.0014197081327438354, 0.0010485743405297399, -0.004372571129351854, 0.0069299908354878426, -0.005105114541947842, -0.003756716148927808, -0.015960371121764183, 0.025554746389389038, 0.003516856813803315, 0.005951105151325464, 0.009736993350088596, 0.043459922075271606, -0.008952588774263859, 0.021315069869160652, -0.011318768374621868, -0.016375262290239334, -0.004560569301247597, -0.026656802743673325, 0.004842565860599279, 0.0004894427256658673, -0.023635871708393097, 0.007448605261743069, -0.008965553715825081, 0.0026092808693647385, -0.01999260112643242, -0.007811635732650757, 0.012142069637775421, -0.01375625841319561, -0.02102983184158802, -0.006806819699704647, 0.015869613736867905, -0.0074032265692949295, -0.001892944099381566, -0.0037016132846474648, -0.005322284530848265, 0.03293204307556152, -0.014430457726120949, 0.0418262854218483, -0.012641236186027527, 0.018216345459222794, -0.028290439397096634, 0.02576219104230404, 0.008433973416686058, 0.013963703997433186, 0.030598275363445282, -0.01225227490067482, 0.012051312252879143, 0.0014553628861904144, -0.008822934702038765, 0.01100111659616232, 0.009860164485871792, -0.004388778004795313, -0.01685498282313347, 0.01091035921126604, -0.00033223762875422835, -0.007850532419979572, -0.0006320617976598442, 0.002114976057782769, -0.007532880175858736, 0.01710132323205471, 0.015610306523740292, -0.009036863222718239, 0.008200597018003464, 0.012174483388662338, 0.00447305291891098, 0.0186053067445755, -0.019253576174378395, 0.010638087056577206, -0.02086128108203411, 0.022404160350561142, 0.010437123477458954, 0.0006920266896486282, -0.02128913812339306, -0.009296170435845852, -0.004106780979782343, 0.044808320701122284, -0.013782189227640629, -0.003750233445316553, -0.01181145291775465, 0.02764216996729374, 0.011960554867982864, -0.005043528974056244, 0.006155309733003378, -0.015584375709295273, 0.012433790601789951, -0.021600307896733284, -0.04314875230193138, -0.01214855257421732, -0.024776823818683624, 0.039077628403902054, 0.016271540895104408, 0.000348039175150916, -0.01511762198060751, 0.0014926382573321462, -0.04068533331155777, -0.0020290804095566273, -0.006904060021042824, 0.02099093608558178, 0.017049461603164673, -0.006981851998716593, 0.007364330347627401, 0.007416191976517439, 0.00766253424808383, 0.02153548039495945, -0.002995000686496496, 0.02157437615096569, -0.011312286369502544, -0.009685131721198559, 0.00414891866967082, -0.009672166779637337, -0.01308854203671217, -0.003380720503628254, -0.003168412484228611, 0.013769223354756832, -0.012615305371582508, 0.007973702624440193, 0.001315985107794404, -0.006139102857559919, -0.028212646022439003, 0.0004906582762487233, 0.0006340876570902765, 0.013289504684507847, -0.010359331034123898, -0.02956104464828968, 0.0263456329703331, 0.02621597982943058, 0.005357939284294844, -0.022754225879907608, -0.009393410757184029, 0.007053161505609751, -0.018086692318320274, -0.0012552099069580436, 0.003977127373218536, -0.010839049704372883, -0.01584368385374546, 0.007753291632980108, 0.005951105151325464, 0.02478978969156742, -0.00858955830335617, 0.007280055433511734, 0.013257091864943504, -0.0000065713156800484285, 0.007234676741063595, -0.00413919473066926, -0.01467679999768734, -0.018333034589886665, -0.017658835276961327, -0.01681608520448208, 0.005108356010168791, -0.007630120497196913, 0.008479352109134197, -0.02771996334195137, 0.004567051772028208, -0.018579376861453056, -0.003983610309660435, -0.0023110774345695972, 0.023065393790602684, 0.04281165450811386, -0.015273206867277622, -0.006696613971143961, 0.002272181212902069, -0.008356180973351002, -0.014508250169456005, -0.0066090975888073444, 0.00827838946133852, -0.016906842589378357, 0.003750233445316553, -0.008524730801582336, -0.0022802846506237984, -0.005156976170837879, -0.009633270092308521, -0.035940006375312805, -0.004323950968682766, 0.027771824970841408, 0.19261354207992554, -0.014547145925462246, -0.006657717749476433, 0.013808120042085648, -0.021340999752283096, 0.011869796551764011, 0.024115590378642082, 0.014080392196774483, 0.0023856281768530607, 0.0005133476224727929, -0.016206713393330574, 0.01723097823560238, 0.008012599311769009, 0.0019723570439964533, 0.006560477428138256, -0.040996503084897995, -0.010657534934580326, 0.00037032339605502784, -0.027875546365976334, -0.011727177537977695, -0.00768198212608695, -0.007299503777176142, -0.011202080175280571, -0.01939619518816471, 0.039622172713279724, -0.011668833903968334, -0.015830717980861664, 0.016919808462262154, 0.03207632899284363, 0.015960371121764183, -0.01093629002571106, -0.016842016950249672, -0.008336733095347881, -0.013244125992059708, -0.011999450623989105, -0.020122256129980087, -0.007422674912959337, -0.02501020021736622, -0.008505282923579216, -0.005526489112526178, -0.0011830900330096483, 0.01773662678897381, 0.010709396563470364, -0.007267090491950512, 0.015999266877770424, 0.02604742906987667, -0.013315435498952866, 0.01621967926621437, -0.02082238532602787, -0.01689387857913971, -0.0439007468521595, -0.03358031064271927, 0.000994281843304634, 0.03726247698068619, -0.02208002656698227, 0.000011990435268671717, 0.006949438713490963, 0.020433424040675163, 0.00515373470261693, -0.031298406422138214, 0.0031116888858377934, 0.015701064839959145, -0.02813485451042652, -0.007377295754849911, 0.007461570668965578, 0.03985555097460747, -0.010975186713039875, -0.025697365403175354, 0.0397258959710598, -0.026319703087210655, -0.0030403793789446354, -0.010067610070109367, -0.002486109733581543, -0.0088812792673707, 0.0017438423819839954, -0.001923736883327365, 0.017827384173870087, 0.006220136769115925, 0.010255607776343822, 0.001199296792037785, -0.01772366091609001, 0.035136155784130096, -0.0061066895723342896, -0.010735327377915382, -0.010651051998138428, -0.026151152327656746, 0.006981851998716593, 0.006622062996029854, -0.010048162192106247, -0.0009124379721470177, -0.00419105589389801, -0.019668467342853546, 0.00012296844215597957, 0.004894427489489317, 0.006852198392152786, 0.010437123477458954, 0.005908967927098274, 0.0038247844204306602, -0.008103356696665287, -0.006456754636019468, -0.028653468936681747, 0.018216345459222794, 0.032205980271101, 0.00022101905778981745, -0.029664767906069756, -0.008155218325555325, 0.03871459513902664, 0.03394334018230438, 0.005860347766429186, -0.013600673526525497, -0.016958704218268394, 0.006372479721903801, 0.0012543996563181281, -0.01911095716059208, -0.010437123477458954, 0.008356180973351002, -0.012855164706707, -0.008472870104014874, 0.019370263442397118, -0.029457321390509605, 0.0034487885423004627, -0.015415825881063938, -0.00047364120837301016, 0.008887761272490025, -0.0020015290938317776, 0.010501950047910213, -0.007500466890633106, -0.0017470837337896228, 0.01717911660671234, -0.024063728749752045, 0.026734594255685806, -0.024556411430239677, 0.0013573121977970004, -0.00010007645323639736, -0.00450546620413661, 0.007513432297855616, 0.027201347053050995, 0.003426099196076393, -0.022183749824762344, 0.002813485451042652, 0.008064460940659046, 0.002243009163066745, 0.009899060241878033, 0.010988151654601097, -0.004790704697370529, -0.004638361278921366, 0.006025656126439571, -0.010605673305690289, -0.01625857502222061, -0.020342666655778885, -0.016090024262666702, -0.026410460472106934, 0.0121226217597723, -0.009406376630067825, 0.0023759042378515005, -0.0273828636854887, -0.015260240994393826, -0.004832841921597719, -0.0007702240254729986, 0.01856641098856926, -0.031039098277688026, 0.0073967440985143185, 0.018721995875239372, -0.023026498034596443, -0.008200597018003464, -0.023480286821722984, -0.16450461745262146, 0.025710329413414, 0.01681608520448208, -0.009023898281157017, 0.023428425192832947, -0.022754225879907608, 0.027616240084171295, 0.015234310179948807, -0.009224860928952694, 0.005166700109839439, 0.0008131718495860696, 0.0038507150020450354, -0.03153178095817566, -0.0026757284067571163, 0.003335341578349471, 0.00672254478558898, -0.030546413734555244, 0.036277107894420624, 0.017256908118724823, 0.0010526260593906045, 0.0053125605918467045, -0.02091314271092415, -0.0016555157490074635, -0.0012454859679564834, 0.023467320948839188, 0.009497134014964104, 0.0046351198107004166, 0.005380628630518913, -0.021691065281629562, -0.013062611222267151, -0.048023734241724014, -0.0008427490829490125, 0.017321735620498657, 0.021340999752283096, 0.011740143410861492, 0.012219862081110477, -0.012984818778932095, 0.007020748220384121, -0.015130587853491306, -0.016193747520446777, 0.0071439193561673164, 0.03236156702041626, 0.024997234344482422, 0.01185683161020279, 0.010735327377915382, 0.04636416584253311, 0.014599007554352283, -0.009004450403153896, 0.019383229315280914, -0.009607339277863503, -0.00414891866967082, -0.008336733095347881, -0.019888877868652344, -0.0005830365116707981, 0.02771996334195137, 0.005620488431304693, -0.00701426574960351, 0.013730327598750591, 0.014145219698548317, 0.011331734247505665, -0.021807754412293434, 0.022857949137687683, 0.01593444123864174, -0.0031343784648925066, 0.001282761339098215, -0.028627539053559303, 0.013354332186281681, 0.0034098925534635782, -0.014689764939248562, -0.004784221760928631, -0.015208380296826363, -0.00796722061932087, -0.008693280629813671, -0.02311725541949272, 0.011629937216639519, -0.012323584407567978, -0.03243935853242874, 0.007643085904419422, 0.00766253424808383, 0.0028702090494334698, -0.017412493005394936, 0.026267841458320618, 0.010884428396821022, -0.03448788449168205, 0.004327192436903715, 0.018838683143258095, -0.02228747308254242, -0.014702730812132359, -0.01020374707877636, -0.027694031596183777, 0.006122896447777748, -0.004252641461789608, -0.012686614878475666, -0.008829417638480663, 0.03319134935736656, 0.01789221167564392, 0.021250242367386818, -0.006683648563921452, 0.009412859566509724, -0.02294870652258396, 0.0009659201023168862, -0.008336733095347881, -0.019603639841079712, -0.012116138823330402, 0.009775889106094837, 0.03993334248661995, 0.009892578236758709, 0.017153184860944748, 0.015545479021966457, -0.01288109552115202, -0.020433424040675163, 0.013652535155415535, 0.022170783951878548, 0.024102624505758286, -0.003623821074143052, 0.03230970352888107, 0.01852751523256302, -0.03132433444261551, -0.017218012362718582, 0.011279872618615627, 0.052250444889068604, 0.005604281555861235, 0.010722361505031586, 0.006155309733003378, -0.016362298280000687, 0.0038020950742065907, -0.1179330125451088, 0.0006563718779943883, 0.006923507899045944, 0.010300987400114536, 0.010313952341675758, 0.016128921881318092, 0.010683465749025345, 0.032050397247076035, -0.0040808506309986115, 0.011318768374621868, -0.021016865968704224, -0.05787741392850876, -0.018786821514368057, -0.03155771270394325, 0.010994634591042995, -0.00672254478558898, 0.010605673305690289, -0.029042430222034454, -0.018294138833880424, 0.01147435326129198, -0.0367957204580307, 0.016906842589378357, 0.010385261848568916, -0.006301170215010643, -0.014547145925462246, -0.004356364719569683, -0.03181701898574829, -0.0031505851075053215, 0.004936564713716507, 0.012952405028045177, 0.010398227721452713, -0.018164483830332756, 0.01794407330453396, -0.020977970212697983, -0.0003221084189135581, 0.007617155089974403, 0.0036173383705317974, -0.01026209071278572, 0.037651438266038895, -0.015662167221307755, 0.011156701482832432, -0.014599007554352283, -0.00417160801589489, -0.03412485495209694, 0.010022231377661228, 0.004084091633558273, -0.005659384187310934, -0.0012973473640158772, 0.02266346849501133, -0.015195414423942566, -0.021211346611380577, -0.01056029461324215, -0.008284871466457844, -0.001761669758707285, 0.005711245816200972, 0.009205413050949574, 0.008997967466711998, -0.021470654755830765, -0.02379145473241806, 0.013872946612536907, -0.004845807328820229, -0.016193747520446777, -0.003912300802767277, 0.00304200011305511, -0.002687073079869151, -0.007617155089974403, 0.003983610309660435, -0.028575677424669266, 0.012939440086483955, -0.004434156697243452, -0.020381562411785126, 0.0030338966753333807, -0.022131888195872307, 0.0012187449028715491, -0.020251909270882607, 0.003640027716755867, -0.018449721857905388, 0.008557144552469254, 0.029042430222034454, -0.00807094294577837, -0.025502884760499, -0.025852948427200317, 0.03057234361767769, -0.032205980271101, 0.011779039166867733, 0.007481019012629986, -0.010112988762557507, 0.017114289104938507, 0.011623455211520195, -0.01751621626317501, -0.009341550059616566, -0.005238009616732597, -0.013561777770519257, -0.005001391749829054, -0.014702730812132359, 0.008609006181359291, -0.009568443521857262, 0.00857011042535305, 0.0027049004565924406, 0.009101689793169498, -0.018294138833880424, -0.0010202126577496529, -0.07286538183689117, 0.01214855257421732, -0.011403043754398823, 0.0032656528055667877, 0.0046221548691391945, -0.017568077892065048, 0.019136887043714523, -0.022546779364347458, -0.0037826469633728266, -0.004913875367492437, -0.01572699472308159, 0.034306369721889496, 0.013509916141629219, -0.014158184640109539, -0.011014082469046116, -0.010787188075482845, 0.005027322564274073, 0.002033942611888051, 0.017218012362718582, 0.02478978969156742, -0.010430640541017056, 0.0027437966782599688, 0.00802556425333023, -0.01406742725521326, 0.0038442325312644243, 0.0035038914065808058, -0.003345065750181675, 0.02386924810707569, -0.00384747376665473, -0.012550478801131248, 0.0008500420954078436, -0.012025381438434124, 0.015662167221307755, 0.019914809614419937, 0.029846282675862312, -0.026656802743673325, -0.006203929893672466, -0.011033530347049236, 0.04947585612535477, 0.03575849160552025, -0.007960737682878971, -0.024984268471598625, 0.013548812828958035, -0.02484164945781231, -0.01406742725521326, 0.011681798845529556, -0.022806087508797646, 0.0018378413515165448, 0.030961304903030396, 0.007928323931992054, 0.03251715004444122, 0.010884428396821022, 0.002808623481541872, -0.008466387167572975, 0.02379145473241806, -0.019162818789482117, 0.05357291176915169, -0.0044049848802387714, -0.0040581608191132545, -0.013471020385622978, 0.020718662068247795, 0.00829783733934164, -0.01685498282313347, -0.004100298509001732, -0.000024132808903232217, -0.014482319355010986, -0.03404706344008446, 0.007876462303102016, 0.0011855211341753602, -0.0405556783080101, -0.01225227490067482, -0.006268756929785013, 0.015312102623283863, 0.015299137681722641, 0.013548812828958035, 0.014560110867023468, 0.01471569575369358, -0.002121458761394024, -0.01100111659616232, 0.00015933225222397596, 0.01965550146996975, -0.003769681556150317, -0.02826450765132904, 0.01413225382566452, 0.029198015108704567, 0.048568278551101685, -0.005461662542074919, 0.014274872839450836, 0.0036529931239783764, 0.005160217639058828, 0.0010145402047783136, 0.017243942245841026, -0.006793854292482138, 0.0005344163510017097, 0.01341915875673294, 0.019538814201951027, -0.003721061395481229, -0.01056029461324215, 0.005033805035054684, 0.03562884032726288, -0.004832841921597719, 0.012997783720493317, -0.006443789228796959, -0.015817752107977867, -0.015947405248880386, 0.023208012804389, -0.019590675830841064, -0.0267605260014534, -0.008045012131333351, 0.0018005658639594913, 0.022935740649700165, -0.006323859561234713, -0.0033677550964057446, 0.02027783915400505, -0.020433424040675163, 0.01689387857913971, -0.027019832283258438, -0.040166717022657394, -0.0001858707400970161, 0.009393410757184029, 0.0010169713059440255, 0.00733839999884367, 0.00923782680183649, -0.011007599532604218, 0.022339332848787308, 0.01406742725521326, 0.01659567467868328, -0.021794788539409637, -0.019085025414824486, -0.008245975710451603, 0.016323402523994446, -0.0023759042378515005, -0.0077208783477544785, -0.001962633104994893, 0.0036108556669205427, -0.0008832658641040325, -0.0267605260014534, 0.02873126231133938, -0.022935740649700165, 0.02412855438888073, -0.005513523705303669, 0.00507918419316411, 0.005951105151325464, 0.005173183046281338, 0.005001391749829054, -0.0177625585347414, 0.015130587853491306, -0.01999260112643242, -0.0192406103014946, 0.02140582725405693, -0.007377295754849911, 0.02927580662071705, -0.015104657039046288, -0.01216800045222044, 0.0034552712459117174, -0.003481202060356736, 0.013561777770519257, 0.0022592158056795597, -0.01183090079575777, 0.013665501028299332, 0.017075393348932266, 0.008511765860021114, -0.007409709505736828, -0.0014261907199397683, -0.003058206755667925, 0.0031424816697835922, 0.0021846650633960962, -0.000989419873803854, -0.022144854068756104, 0.0036270625423640013, -0.006233102176338434, -0.016504917293787003, 0.003999816719442606, 0.017931107431650162, -0.025956671684980392, 0.021626237779855728, 0.002192768268287182, -0.001260882243514061, 0.018721995875239372, -0.023493250831961632, 0.030598275363445282, -0.011616972275078297, -0.019331367686390877, 0.01505279541015625, 0.01505279541015625, 0.004524914547801018, 0.0019042887724936008, -0.014962038025259972]
```

Words that are semantically similar are often represented by vectors that are close to each other in this vector space. This allows for mathematical operations like addition and subtraction to carry semantic meaning. For example, the vector representation of "king" minus "man" plus "woman" should be close to the vector representation of "queen." In other words, vector embeddings are a numerical representation of a particular data object.

A vector index allows you to retrieve a neighborhood of nodes or relationships based on the similarity between the embedding properties of those nodes or relationships and the ones specified in the query.---
title: "rig - Rust"
source: "https://docs.rs/rig-core/latest/rig/#macros"
author:
published:
created: 2024-12-23
description: "Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity."
tags:
  - "clippings"
---
Expand description

Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.

## §Table of contents

- [High-level features](https://docs.rs/rig-core/latest/rig/#high-level-features)
- [Simple Example](https://docs.rs/rig-core/latest/rig/#simple-example)
- [Core Concepts](https://docs.rs/rig-core/latest/rig/#core-concepts)
- [Integrations](https://docs.rs/rig-core/latest/rig/#integrations)

## §High-level features

- Full support for LLM completion and embedding workflows
- Simple but powerful common abstractions over LLM providers (e.g. OpenAI, Cohere) and vector stores (e.g. MongoDB, in-memory)
- Integrate LLMs in your app with minimal boilerplate

## §Simple example:

```
use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and agent.
    // This requires the \`OPENAI_API_KEY\` environment variable to be set.
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.agent("gpt-4").build();

    // Prompt the model and print its response
    let response = gpt4
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
```

Note: using `#[tokio::main]` requires you enable tokio’s `macros` and `rt-multi-thread` features or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

## §Core concepts
### §Completion and embedding models

Rig provides a consistent API for working with LLMs and embeddings. Specifically, each provider (e.g. OpenAI, Cohere) has a `Client` struct that can be used to initialize completion and embedding models. These models implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits respectively, which provide a common, low-level interface for creating completion and embedding requests and executing them.

### §Agents

Rig also provides high-level abstractions over LLMs in the form of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type.

The [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type can be used to create anything from simple agents that use vanilla models to full blown RAG systems that can be used to answer questions using a knowledge base.

### §Vector stores and indexes

Rig provides a common interface for working with vector stores and indexes. Specifically, the library provides the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait, which can be implemented to define vector stores and indices respectively. Those can then be used as the knowledge base for a RAG enabled [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent"), or as a source of context documents in a custom architecture that use multiple LLMs or agents.

## §Integrations
### §Model Providers

Rig natively supports the following completion and embedding model provider integrations:

- OpenAI
- Cohere
- Anthropic
- Perplexity
- Gemini

You can also implement your own model provider integration by defining types that implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits.

### §Vector Stores

Rig currently supports the following vector store integrations via companion crates:

- `rig-mongodb`: Vector store implementation for MongoDB
- `rig-lancedb`: Vector store implementation for LanceDB
- `rig-neo4j`: Vector store implementation for Neo4j
- `rig-qdrant`: Vector store implementation for Qdrant

You can also implement your own vector store integration by defining types that implement the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait.

- `pub use embeddings::[Embed](https://docs.rs/rig-core/latest/rig/embeddings/embed/trait.Embed.html "trait rig::embeddings::embed::Embed");`

- This module contains the implementation of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") struct and its builder.

- This module provides functionality for working with completion models. It provides traits, structs, and enums for generating completion requests, handling completion responses, and defining completion models.
- This module provides functionality for working with embeddings. Embeddings are numerical representations of documents or other objects, typically used in natural language processing (NLP) tasks such as text classification, information retrieval, and document similarity.
- This module provides high-level abstractions for extracting structured data from text using LLMs.
- This module provides utility structs for loading and preprocessing files.

- This module defines a flexible pipeline API for defining a sequence of operations that may or may not use AI components (e.g.: semantic search, LLMs prompting, etc).
- This module contains clients for the different LLM providers that Rig supports.
- Module defining tool related structs and traits.---
title: "rig - Rust"
source: "https://docs.rs/rig-core/latest/rig/#vector-stores"
author:
published:
created: 2024-12-23
description: "Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity."
tags:
  - "clippings"
---
Expand description

Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.

## §Table of contents

- [High-level features](https://docs.rs/rig-core/latest/rig/#high-level-features)
- [Simple Example](https://docs.rs/rig-core/latest/rig/#simple-example)
- [Core Concepts](https://docs.rs/rig-core/latest/rig/#core-concepts)
- [Integrations](https://docs.rs/rig-core/latest/rig/#integrations)

## §High-level features

- Full support for LLM completion and embedding workflows
- Simple but powerful common abstractions over LLM providers (e.g. OpenAI, Cohere) and vector stores (e.g. MongoDB, in-memory)
- Integrate LLMs in your app with minimal boilerplate

## §Simple example:

```
use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and agent.
    // This requires the \`OPENAI_API_KEY\` environment variable to be set.
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.agent("gpt-4").build();

    // Prompt the model and print its response
    let response = gpt4
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
```

Note: using `#[tokio::main]` requires you enable tokio’s `macros` and `rt-multi-thread` features or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

## §Core concepts
### §Completion and embedding models

Rig provides a consistent API for working with LLMs and embeddings. Specifically, each provider (e.g. OpenAI, Cohere) has a `Client` struct that can be used to initialize completion and embedding models. These models implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits respectively, which provide a common, low-level interface for creating completion and embedding requests and executing them.

### §Agents

Rig also provides high-level abstractions over LLMs in the form of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type.

The [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type can be used to create anything from simple agents that use vanilla models to full blown RAG systems that can be used to answer questions using a knowledge base.

### §Vector stores and indexes

Rig provides a common interface for working with vector stores and indexes. Specifically, the library provides the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait, which can be implemented to define vector stores and indices respectively. Those can then be used as the knowledge base for a RAG enabled [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent"), or as a source of context documents in a custom architecture that use multiple LLMs or agents.

## §Integrations
### §Model Providers

Rig natively supports the following completion and embedding model provider integrations:

- OpenAI
- Cohere
- Anthropic
- Perplexity
- Gemini

You can also implement your own model provider integration by defining types that implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits.

### §Vector Stores

Rig currently supports the following vector store integrations via companion crates:

- `rig-mongodb`: Vector store implementation for MongoDB
- `rig-lancedb`: Vector store implementation for LanceDB
- `rig-neo4j`: Vector store implementation for Neo4j
- `rig-qdrant`: Vector store implementation for Qdrant

You can also implement your own vector store integration by defining types that implement the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait.

- `pub use embeddings::[Embed](https://docs.rs/rig-core/latest/rig/embeddings/embed/trait.Embed.html "trait rig::embeddings::embed::Embed");`

- This module contains the implementation of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") struct and its builder.

- This module provides functionality for working with completion models. It provides traits, structs, and enums for generating completion requests, handling completion responses, and defining completion models.
- This module provides functionality for working with embeddings. Embeddings are numerical representations of documents or other objects, typically used in natural language processing (NLP) tasks such as text classification, information retrieval, and document similarity.
- This module provides high-level abstractions for extracting structured data from text using LLMs.
- This module provides utility structs for loading and preprocessing files.

- This module defines a flexible pipeline API for defining a sequence of operations that may or may not use AI components (e.g.: semantic search, LLMs prompting, etc).
- This module contains clients for the different LLM providers that Rig supports.
- Module defining tool related structs and traits.---
title: "rig - Rust"
source: "https://docs.rs/rig-core/latest/rig/#reexports"
author:
published:
created: 2024-12-23
description: "Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity."
tags:
  - "clippings"
---
Expand description

Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.

## §Table of contents

- [High-level features](https://docs.rs/rig-core/latest/rig/#high-level-features)
- [Simple Example](https://docs.rs/rig-core/latest/rig/#simple-example)
- [Core Concepts](https://docs.rs/rig-core/latest/rig/#core-concepts)
- [Integrations](https://docs.rs/rig-core/latest/rig/#integrations)

## §High-level features

- Full support for LLM completion and embedding workflows
- Simple but powerful common abstractions over LLM providers (e.g. OpenAI, Cohere) and vector stores (e.g. MongoDB, in-memory)
- Integrate LLMs in your app with minimal boilerplate

## §Simple example:

```
use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and agent.
    // This requires the \`OPENAI_API_KEY\` environment variable to be set.
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.agent("gpt-4").build();

    // Prompt the model and print its response
    let response = gpt4
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
```

Note: using `#[tokio::main]` requires you enable tokio’s `macros` and `rt-multi-thread` features or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

## §Core concepts
### §Completion and embedding models

Rig provides a consistent API for working with LLMs and embeddings. Specifically, each provider (e.g. OpenAI, Cohere) has a `Client` struct that can be used to initialize completion and embedding models. These models implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits respectively, which provide a common, low-level interface for creating completion and embedding requests and executing them.

### §Agents

Rig also provides high-level abstractions over LLMs in the form of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type.

The [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type can be used to create anything from simple agents that use vanilla models to full blown RAG systems that can be used to answer questions using a knowledge base.

### §Vector stores and indexes

Rig provides a common interface for working with vector stores and indexes. Specifically, the library provides the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait, which can be implemented to define vector stores and indices respectively. Those can then be used as the knowledge base for a RAG enabled [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent"), or as a source of context documents in a custom architecture that use multiple LLMs or agents.

## §Integrations
### §Model Providers

Rig natively supports the following completion and embedding model provider integrations:

- OpenAI
- Cohere
- Anthropic
- Perplexity
- Gemini

You can also implement your own model provider integration by defining types that implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits.

### §Vector Stores

Rig currently supports the following vector store integrations via companion crates:

- `rig-mongodb`: Vector store implementation for MongoDB
- `rig-lancedb`: Vector store implementation for LanceDB
- `rig-neo4j`: Vector store implementation for Neo4j
- `rig-qdrant`: Vector store implementation for Qdrant

You can also implement your own vector store integration by defining types that implement the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait.

- `pub use embeddings::[Embed](https://docs.rs/rig-core/latest/rig/embeddings/embed/trait.Embed.html "trait rig::embeddings::embed::Embed");`

- This module contains the implementation of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") struct and its builder.

- This module provides functionality for working with completion models. It provides traits, structs, and enums for generating completion requests, handling completion responses, and defining completion models.
- This module provides functionality for working with embeddings. Embeddings are numerical representations of documents or other objects, typically used in natural language processing (NLP) tasks such as text classification, information retrieval, and document similarity.
- This module provides high-level abstractions for extracting structured data from text using LLMs.
- This module provides utility structs for loading and preprocessing files.

- This module defines a flexible pipeline API for defining a sequence of operations that may or may not use AI components (e.g.: semantic search, LLMs prompting, etc).
- This module contains clients for the different LLM providers that Rig supports.
- Module defining tool related structs and traits.---
title: "rig - Rust"
source: "https://docs.rs/rig-core/latest/rig/#modules"
author:
published:
created: 2024-12-23
description: "Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity."
tags:
  - "clippings"
---
Expand description

Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.

## §Table of contents

- [High-level features](https://docs.rs/rig-core/latest/rig/#high-level-features)
- [Simple Example](https://docs.rs/rig-core/latest/rig/#simple-example)
- [Core Concepts](https://docs.rs/rig-core/latest/rig/#core-concepts)
- [Integrations](https://docs.rs/rig-core/latest/rig/#integrations)

## §High-level features

- Full support for LLM completion and embedding workflows
- Simple but powerful common abstractions over LLM providers (e.g. OpenAI, Cohere) and vector stores (e.g. MongoDB, in-memory)
- Integrate LLMs in your app with minimal boilerplate

## §Simple example:

```
use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and agent.
    // This requires the \`OPENAI_API_KEY\` environment variable to be set.
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.agent("gpt-4").build();

    // Prompt the model and print its response
    let response = gpt4
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
```

Note: using `#[tokio::main]` requires you enable tokio’s `macros` and `rt-multi-thread` features or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

## §Core concepts
### §Completion and embedding models

Rig provides a consistent API for working with LLMs and embeddings. Specifically, each provider (e.g. OpenAI, Cohere) has a `Client` struct that can be used to initialize completion and embedding models. These models implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits respectively, which provide a common, low-level interface for creating completion and embedding requests and executing them.

### §Agents

Rig also provides high-level abstractions over LLMs in the form of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type.

The [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type can be used to create anything from simple agents that use vanilla models to full blown RAG systems that can be used to answer questions using a knowledge base.

### §Vector stores and indexes

Rig provides a common interface for working with vector stores and indexes. Specifically, the library provides the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait, which can be implemented to define vector stores and indices respectively. Those can then be used as the knowledge base for a RAG enabled [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent"), or as a source of context documents in a custom architecture that use multiple LLMs or agents.

## §Integrations
### §Model Providers

Rig natively supports the following completion and embedding model provider integrations:

- OpenAI
- Cohere
- Anthropic
- Perplexity
- Gemini

You can also implement your own model provider integration by defining types that implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits.

### §Vector Stores

Rig currently supports the following vector store integrations via companion crates:

- `rig-mongodb`: Vector store implementation for MongoDB
- `rig-lancedb`: Vector store implementation for LanceDB
- `rig-neo4j`: Vector store implementation for Neo4j
- `rig-qdrant`: Vector store implementation for Qdrant

You can also implement your own vector store integration by defining types that implement the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait.

- `pub use embeddings::[Embed](https://docs.rs/rig-core/latest/rig/embeddings/embed/trait.Embed.html "trait rig::embeddings::embed::Embed");`

- This module contains the implementation of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") struct and its builder.

- This module provides functionality for working with completion models. It provides traits, structs, and enums for generating completion requests, handling completion responses, and defining completion models.
- This module provides functionality for working with embeddings. Embeddings are numerical representations of documents or other objects, typically used in natural language processing (NLP) tasks such as text classification, information retrieval, and document similarity.
- This module provides high-level abstractions for extracting structured data from text using LLMs.
- This module provides utility structs for loading and preprocessing files.

- This module defines a flexible pipeline API for defining a sequence of operations that may or may not use AI components (e.g.: semantic search, LLMs prompting, etc).
- This module contains clients for the different LLM providers that Rig supports.
- Module defining tool related structs and traits.---
title: "rig - Rust"
source: "https://docs.rs/rig-core/latest/rig/#table-of-contents"
author:
published:
created: 2024-12-23
description: "Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity."
tags:
  - "clippings"
---
Expand description

Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.

## §Table of contents

- [High-level features](https://docs.rs/rig-core/latest/rig/#high-level-features)
- [Simple Example](https://docs.rs/rig-core/latest/rig/#simple-example)
- [Core Concepts](https://docs.rs/rig-core/latest/rig/#core-concepts)
- [Integrations](https://docs.rs/rig-core/latest/rig/#integrations)

## §High-level features

- Full support for LLM completion and embedding workflows
- Simple but powerful common abstractions over LLM providers (e.g. OpenAI, Cohere) and vector stores (e.g. MongoDB, in-memory)
- Integrate LLMs in your app with minimal boilerplate

## §Simple example:

```
use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and agent.
    // This requires the \`OPENAI_API_KEY\` environment variable to be set.
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.agent("gpt-4").build();

    // Prompt the model and print its response
    let response = gpt4
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
```

Note: using `#[tokio::main]` requires you enable tokio’s `macros` and `rt-multi-thread` features or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

## §Core concepts
### §Completion and embedding models

Rig provides a consistent API for working with LLMs and embeddings. Specifically, each provider (e.g. OpenAI, Cohere) has a `Client` struct that can be used to initialize completion and embedding models. These models implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits respectively, which provide a common, low-level interface for creating completion and embedding requests and executing them.

### §Agents

Rig also provides high-level abstractions over LLMs in the form of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type.

The [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type can be used to create anything from simple agents that use vanilla models to full blown RAG systems that can be used to answer questions using a knowledge base.

### §Vector stores and indexes

Rig provides a common interface for working with vector stores and indexes. Specifically, the library provides the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait, which can be implemented to define vector stores and indices respectively. Those can then be used as the knowledge base for a RAG enabled [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent"), or as a source of context documents in a custom architecture that use multiple LLMs or agents.

## §Integrations
### §Model Providers

Rig natively supports the following completion and embedding model provider integrations:

- OpenAI
- Cohere
- Anthropic
- Perplexity
- Gemini

You can also implement your own model provider integration by defining types that implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits.

### §Vector Stores

Rig currently supports the following vector store integrations via companion crates:

- `rig-mongodb`: Vector store implementation for MongoDB
- `rig-lancedb`: Vector store implementation for LanceDB
- `rig-neo4j`: Vector store implementation for Neo4j
- `rig-qdrant`: Vector store implementation for Qdrant

You can also implement your own vector store integration by defining types that implement the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait.

- `pub use embeddings::[Embed](https://docs.rs/rig-core/latest/rig/embeddings/embed/trait.Embed.html "trait rig::embeddings::embed::Embed");`

- This module contains the implementation of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") struct and its builder.

- This module provides functionality for working with completion models. It provides traits, structs, and enums for generating completion requests, handling completion responses, and defining completion models.
- This module provides functionality for working with embeddings. Embeddings are numerical representations of documents or other objects, typically used in natural language processing (NLP) tasks such as text classification, information retrieval, and document similarity.
- This module provides high-level abstractions for extracting structured data from text using LLMs.
- This module provides utility structs for loading and preprocessing files.

- This module defines a flexible pipeline API for defining a sequence of operations that may or may not use AI components (e.g.: semantic search, LLMs prompting, etc).
- This module contains clients for the different LLM providers that Rig supports.
- Module defining tool related structs and traits.---
title: "rig - Rust"
source: "https://docs.rs/rig-core/latest/rig/#high-level-features"
author:
published:
created: 2024-12-23
description: "Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity."
tags:
  - "clippings"
---
Expand description

Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.

## §Table of contents

- [High-level features](https://docs.rs/rig-core/latest/rig/#high-level-features)
- [Simple Example](https://docs.rs/rig-core/latest/rig/#simple-example)
- [Core Concepts](https://docs.rs/rig-core/latest/rig/#core-concepts)
- [Integrations](https://docs.rs/rig-core/latest/rig/#integrations)

## §High-level features

- Full support for LLM completion and embedding workflows
- Simple but powerful common abstractions over LLM providers (e.g. OpenAI, Cohere) and vector stores (e.g. MongoDB, in-memory)
- Integrate LLMs in your app with minimal boilerplate

## §Simple example:

```
use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and agent.
    // This requires the \`OPENAI_API_KEY\` environment variable to be set.
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.agent("gpt-4").build();

    // Prompt the model and print its response
    let response = gpt4
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
```

Note: using `#[tokio::main]` requires you enable tokio’s `macros` and `rt-multi-thread` features or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

## §Core concepts
### §Completion and embedding models

Rig provides a consistent API for working with LLMs and embeddings. Specifically, each provider (e.g. OpenAI, Cohere) has a `Client` struct that can be used to initialize completion and embedding models. These models implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits respectively, which provide a common, low-level interface for creating completion and embedding requests and executing them.

### §Agents

Rig also provides high-level abstractions over LLMs in the form of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type.

The [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type can be used to create anything from simple agents that use vanilla models to full blown RAG systems that can be used to answer questions using a knowledge base.

### §Vector stores and indexes

Rig provides a common interface for working with vector stores and indexes. Specifically, the library provides the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait, which can be implemented to define vector stores and indices respectively. Those can then be used as the knowledge base for a RAG enabled [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent"), or as a source of context documents in a custom architecture that use multiple LLMs or agents.

## §Integrations
### §Model Providers

Rig natively supports the following completion and embedding model provider integrations:

- OpenAI
- Cohere
- Anthropic
- Perplexity
- Gemini

You can also implement your own model provider integration by defining types that implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits.

### §Vector Stores

Rig currently supports the following vector store integrations via companion crates:

- `rig-mongodb`: Vector store implementation for MongoDB
- `rig-lancedb`: Vector store implementation for LanceDB
- `rig-neo4j`: Vector store implementation for Neo4j
- `rig-qdrant`: Vector store implementation for Qdrant

You can also implement your own vector store integration by defining types that implement the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait.

- `pub use embeddings::[Embed](https://docs.rs/rig-core/latest/rig/embeddings/embed/trait.Embed.html "trait rig::embeddings::embed::Embed");`

- This module contains the implementation of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") struct and its builder.

- This module provides functionality for working with completion models. It provides traits, structs, and enums for generating completion requests, handling completion responses, and defining completion models.
- This module provides functionality for working with embeddings. Embeddings are numerical representations of documents or other objects, typically used in natural language processing (NLP) tasks such as text classification, information retrieval, and document similarity.
- This module provides high-level abstractions for extracting structured data from text using LLMs.
- This module provides utility structs for loading and preprocessing files.

- This module defines a flexible pipeline API for defining a sequence of operations that may or may not use AI components (e.g.: semantic search, LLMs prompting, etc).
- This module contains clients for the different LLM providers that Rig supports.
- Module defining tool related structs and traits.---
title: "rig - Rust"
source: "https://docs.rs/rig-core/latest/rig/#simple-example"
author:
published:
created: 2024-12-23
description: "Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity."
tags:
  - "clippings"
---
Expand description

Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.

## §Table of contents

- [High-level features](https://docs.rs/rig-core/latest/rig/#high-level-features)
- [Simple Example](https://docs.rs/rig-core/latest/rig/#simple-example)
- [Core Concepts](https://docs.rs/rig-core/latest/rig/#core-concepts)
- [Integrations](https://docs.rs/rig-core/latest/rig/#integrations)

## §High-level features

- Full support for LLM completion and embedding workflows
- Simple but powerful common abstractions over LLM providers (e.g. OpenAI, Cohere) and vector stores (e.g. MongoDB, in-memory)
- Integrate LLMs in your app with minimal boilerplate

## §Simple example:

```
use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and agent.
    // This requires the \`OPENAI_API_KEY\` environment variable to be set.
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.agent("gpt-4").build();

    // Prompt the model and print its response
    let response = gpt4
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
```

Note: using `#[tokio::main]` requires you enable tokio’s `macros` and `rt-multi-thread` features or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

## §Core concepts
### §Completion and embedding models

Rig provides a consistent API for working with LLMs and embeddings. Specifically, each provider (e.g. OpenAI, Cohere) has a `Client` struct that can be used to initialize completion and embedding models. These models implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits respectively, which provide a common, low-level interface for creating completion and embedding requests and executing them.

### §Agents

Rig also provides high-level abstractions over LLMs in the form of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type.

The [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type can be used to create anything from simple agents that use vanilla models to full blown RAG systems that can be used to answer questions using a knowledge base.

### §Vector stores and indexes

Rig provides a common interface for working with vector stores and indexes. Specifically, the library provides the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait, which can be implemented to define vector stores and indices respectively. Those can then be used as the knowledge base for a RAG enabled [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent"), or as a source of context documents in a custom architecture that use multiple LLMs or agents.

## §Integrations
### §Model Providers

Rig natively supports the following completion and embedding model provider integrations:

- OpenAI
- Cohere
- Anthropic
- Perplexity
- Gemini

You can also implement your own model provider integration by defining types that implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits.

### §Vector Stores

Rig currently supports the following vector store integrations via companion crates:

- `rig-mongodb`: Vector store implementation for MongoDB
- `rig-lancedb`: Vector store implementation for LanceDB
- `rig-neo4j`: Vector store implementation for Neo4j
- `rig-qdrant`: Vector store implementation for Qdrant

You can also implement your own vector store integration by defining types that implement the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait.

- `pub use embeddings::[Embed](https://docs.rs/rig-core/latest/rig/embeddings/embed/trait.Embed.html "trait rig::embeddings::embed::Embed");`

- This module contains the implementation of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") struct and its builder.

- This module provides functionality for working with completion models. It provides traits, structs, and enums for generating completion requests, handling completion responses, and defining completion models.
- This module provides functionality for working with embeddings. Embeddings are numerical representations of documents or other objects, typically used in natural language processing (NLP) tasks such as text classification, information retrieval, and document similarity.
- This module provides high-level abstractions for extracting structured data from text using LLMs.
- This module provides utility structs for loading and preprocessing files.

- This module defines a flexible pipeline API for defining a sequence of operations that may or may not use AI components (e.g.: semantic search, LLMs prompting, etc).
- This module contains clients for the different LLM providers that Rig supports.
- Module defining tool related structs and traits.---
title: "rig - Rust"
source: "https://docs.rs/rig-core/latest/rig/#completion-and-embedding-models"
author:
published:
created: 2024-12-23
description: "Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity."
tags:
  - "clippings"
---
Expand description

Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.

## §Table of contents

- [High-level features](https://docs.rs/rig-core/latest/rig/#high-level-features)
- [Simple Example](https://docs.rs/rig-core/latest/rig/#simple-example)
- [Core Concepts](https://docs.rs/rig-core/latest/rig/#core-concepts)
- [Integrations](https://docs.rs/rig-core/latest/rig/#integrations)

## §High-level features

- Full support for LLM completion and embedding workflows
- Simple but powerful common abstractions over LLM providers (e.g. OpenAI, Cohere) and vector stores (e.g. MongoDB, in-memory)
- Integrate LLMs in your app with minimal boilerplate

## §Simple example:

```
use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and agent.
    // This requires the \`OPENAI_API_KEY\` environment variable to be set.
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.agent("gpt-4").build();

    // Prompt the model and print its response
    let response = gpt4
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
```

Note: using `#[tokio::main]` requires you enable tokio’s `macros` and `rt-multi-thread` features or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

## §Core concepts
### §Completion and embedding models

Rig provides a consistent API for working with LLMs and embeddings. Specifically, each provider (e.g. OpenAI, Cohere) has a `Client` struct that can be used to initialize completion and embedding models. These models implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits respectively, which provide a common, low-level interface for creating completion and embedding requests and executing them.

### §Agents

Rig also provides high-level abstractions over LLMs in the form of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type.

The [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type can be used to create anything from simple agents that use vanilla models to full blown RAG systems that can be used to answer questions using a knowledge base.

### §Vector stores and indexes

Rig provides a common interface for working with vector stores and indexes. Specifically, the library provides the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait, which can be implemented to define vector stores and indices respectively. Those can then be used as the knowledge base for a RAG enabled [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent"), or as a source of context documents in a custom architecture that use multiple LLMs or agents.

## §Integrations
### §Model Providers

Rig natively supports the following completion and embedding model provider integrations:

- OpenAI
- Cohere
- Anthropic
- Perplexity
- Gemini

You can also implement your own model provider integration by defining types that implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits.

### §Vector Stores

Rig currently supports the following vector store integrations via companion crates:

- `rig-mongodb`: Vector store implementation for MongoDB
- `rig-lancedb`: Vector store implementation for LanceDB
- `rig-neo4j`: Vector store implementation for Neo4j
- `rig-qdrant`: Vector store implementation for Qdrant

You can also implement your own vector store integration by defining types that implement the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait.

- `pub use embeddings::[Embed](https://docs.rs/rig-core/latest/rig/embeddings/embed/trait.Embed.html "trait rig::embeddings::embed::Embed");`

- This module contains the implementation of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") struct and its builder.

- This module provides functionality for working with completion models. It provides traits, structs, and enums for generating completion requests, handling completion responses, and defining completion models.
- This module provides functionality for working with embeddings. Embeddings are numerical representations of documents or other objects, typically used in natural language processing (NLP) tasks such as text classification, information retrieval, and document similarity.
- This module provides high-level abstractions for extracting structured data from text using LLMs.
- This module provides utility structs for loading and preprocessing files.

- This module defines a flexible pipeline API for defining a sequence of operations that may or may not use AI components (e.g.: semantic search, LLMs prompting, etc).
- This module contains clients for the different LLM providers that Rig supports.
- Module defining tool related structs and traits.---
title: "rig - Rust"
source: "https://docs.rs/rig-core/latest/rig/#agents"
author:
published:
created: 2024-12-23
description: "Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity."
tags:
  - "clippings"
---
Expand description

Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.

## §Table of contents

- [High-level features](https://docs.rs/rig-core/latest/rig/#high-level-features)
- [Simple Example](https://docs.rs/rig-core/latest/rig/#simple-example)
- [Core Concepts](https://docs.rs/rig-core/latest/rig/#core-concepts)
- [Integrations](https://docs.rs/rig-core/latest/rig/#integrations)

## §High-level features

- Full support for LLM completion and embedding workflows
- Simple but powerful common abstractions over LLM providers (e.g. OpenAI, Cohere) and vector stores (e.g. MongoDB, in-memory)
- Integrate LLMs in your app with minimal boilerplate

## §Simple example:

```
use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and agent.
    // This requires the \`OPENAI_API_KEY\` environment variable to be set.
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.agent("gpt-4").build();

    // Prompt the model and print its response
    let response = gpt4
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
```

Note: using `#[tokio::main]` requires you enable tokio’s `macros` and `rt-multi-thread` features or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

## §Core concepts
### §Completion and embedding models

Rig provides a consistent API for working with LLMs and embeddings. Specifically, each provider (e.g. OpenAI, Cohere) has a `Client` struct that can be used to initialize completion and embedding models. These models implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits respectively, which provide a common, low-level interface for creating completion and embedding requests and executing them.

### §Agents

Rig also provides high-level abstractions over LLMs in the form of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type.

The [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type can be used to create anything from simple agents that use vanilla models to full blown RAG systems that can be used to answer questions using a knowledge base.

### §Vector stores and indexes

Rig provides a common interface for working with vector stores and indexes. Specifically, the library provides the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait, which can be implemented to define vector stores and indices respectively. Those can then be used as the knowledge base for a RAG enabled [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent"), or as a source of context documents in a custom architecture that use multiple LLMs or agents.

## §Integrations
### §Model Providers

Rig natively supports the following completion and embedding model provider integrations:

- OpenAI
- Cohere
- Anthropic
- Perplexity
- Gemini

You can also implement your own model provider integration by defining types that implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits.

### §Vector Stores

Rig currently supports the following vector store integrations via companion crates:

- `rig-mongodb`: Vector store implementation for MongoDB
- `rig-lancedb`: Vector store implementation for LanceDB
- `rig-neo4j`: Vector store implementation for Neo4j
- `rig-qdrant`: Vector store implementation for Qdrant

You can also implement your own vector store integration by defining types that implement the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait.

- `pub use embeddings::[Embed](https://docs.rs/rig-core/latest/rig/embeddings/embed/trait.Embed.html "trait rig::embeddings::embed::Embed");`

- This module contains the implementation of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") struct and its builder.

- This module provides functionality for working with completion models. It provides traits, structs, and enums for generating completion requests, handling completion responses, and defining completion models.
- This module provides functionality for working with embeddings. Embeddings are numerical representations of documents or other objects, typically used in natural language processing (NLP) tasks such as text classification, information retrieval, and document similarity.
- This module provides high-level abstractions for extracting structured data from text using LLMs.
- This module provides utility structs for loading and preprocessing files.

- This module defines a flexible pipeline API for defining a sequence of operations that may or may not use AI components (e.g.: semantic search, LLMs prompting, etc).
- This module contains clients for the different LLM providers that Rig supports.
- Module defining tool related structs and traits.---
title: "rig - Rust"
source: "https://docs.rs/rig-core/latest/rig/#vector-stores-and-indexes"
author:
published:
created: 2024-12-23
description: "Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity."
tags:
  - "clippings"
---
Expand description

Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.

## §Table of contents

- [High-level features](https://docs.rs/rig-core/latest/rig/#high-level-features)
- [Simple Example](https://docs.rs/rig-core/latest/rig/#simple-example)
- [Core Concepts](https://docs.rs/rig-core/latest/rig/#core-concepts)
- [Integrations](https://docs.rs/rig-core/latest/rig/#integrations)

## §High-level features

- Full support for LLM completion and embedding workflows
- Simple but powerful common abstractions over LLM providers (e.g. OpenAI, Cohere) and vector stores (e.g. MongoDB, in-memory)
- Integrate LLMs in your app with minimal boilerplate

## §Simple example:

```
use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and agent.
    // This requires the \`OPENAI_API_KEY\` environment variable to be set.
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.agent("gpt-4").build();

    // Prompt the model and print its response
    let response = gpt4
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
```

Note: using `#[tokio::main]` requires you enable tokio’s `macros` and `rt-multi-thread` features or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

## §Core concepts
### §Completion and embedding models

Rig provides a consistent API for working with LLMs and embeddings. Specifically, each provider (e.g. OpenAI, Cohere) has a `Client` struct that can be used to initialize completion and embedding models. These models implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits respectively, which provide a common, low-level interface for creating completion and embedding requests and executing them.

### §Agents

Rig also provides high-level abstractions over LLMs in the form of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type.

The [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type can be used to create anything from simple agents that use vanilla models to full blown RAG systems that can be used to answer questions using a knowledge base.

### §Vector stores and indexes

Rig provides a common interface for working with vector stores and indexes. Specifically, the library provides the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait, which can be implemented to define vector stores and indices respectively. Those can then be used as the knowledge base for a RAG enabled [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent"), or as a source of context documents in a custom architecture that use multiple LLMs or agents.

## §Integrations
### §Model Providers

Rig natively supports the following completion and embedding model provider integrations:

- OpenAI
- Cohere
- Anthropic
- Perplexity
- Gemini

You can also implement your own model provider integration by defining types that implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits.

### §Vector Stores

Rig currently supports the following vector store integrations via companion crates:

- `rig-mongodb`: Vector store implementation for MongoDB
- `rig-lancedb`: Vector store implementation for LanceDB
- `rig-neo4j`: Vector store implementation for Neo4j
- `rig-qdrant`: Vector store implementation for Qdrant

You can also implement your own vector store integration by defining types that implement the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait.

- `pub use embeddings::[Embed](https://docs.rs/rig-core/latest/rig/embeddings/embed/trait.Embed.html "trait rig::embeddings::embed::Embed");`

- This module contains the implementation of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") struct and its builder.

- This module provides functionality for working with completion models. It provides traits, structs, and enums for generating completion requests, handling completion responses, and defining completion models.
- This module provides functionality for working with embeddings. Embeddings are numerical representations of documents or other objects, typically used in natural language processing (NLP) tasks such as text classification, information retrieval, and document similarity.
- This module provides high-level abstractions for extracting structured data from text using LLMs.
- This module provides utility structs for loading and preprocessing files.

- This module defines a flexible pipeline API for defining a sequence of operations that may or may not use AI components (e.g.: semantic search, LLMs prompting, etc).
- This module contains clients for the different LLM providers that Rig supports.
- Module defining tool related structs and traits.---
title: "rig - Rust"
source: "https://docs.rs/rig-core/latest/rig/#integrations"
author:
published:
created: 2024-12-23
description: "Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity."
tags:
  - "clippings"
---
Expand description

Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.

## §Table of contents

- [High-level features](https://docs.rs/rig-core/latest/rig/#high-level-features)
- [Simple Example](https://docs.rs/rig-core/latest/rig/#simple-example)
- [Core Concepts](https://docs.rs/rig-core/latest/rig/#core-concepts)
- [Integrations](https://docs.rs/rig-core/latest/rig/#integrations)

## §High-level features

- Full support for LLM completion and embedding workflows
- Simple but powerful common abstractions over LLM providers (e.g. OpenAI, Cohere) and vector stores (e.g. MongoDB, in-memory)
- Integrate LLMs in your app with minimal boilerplate

## §Simple example:

```
use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and agent.
    // This requires the \`OPENAI_API_KEY\` environment variable to be set.
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.agent("gpt-4").build();

    // Prompt the model and print its response
    let response = gpt4
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
```

Note: using `#[tokio::main]` requires you enable tokio’s `macros` and `rt-multi-thread` features or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

## §Core concepts
### §Completion and embedding models

Rig provides a consistent API for working with LLMs and embeddings. Specifically, each provider (e.g. OpenAI, Cohere) has a `Client` struct that can be used to initialize completion and embedding models. These models implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits respectively, which provide a common, low-level interface for creating completion and embedding requests and executing them.

### §Agents

Rig also provides high-level abstractions over LLMs in the form of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type.

The [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type can be used to create anything from simple agents that use vanilla models to full blown RAG systems that can be used to answer questions using a knowledge base.

### §Vector stores and indexes

Rig provides a common interface for working with vector stores and indexes. Specifically, the library provides the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait, which can be implemented to define vector stores and indices respectively. Those can then be used as the knowledge base for a RAG enabled [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent"), or as a source of context documents in a custom architecture that use multiple LLMs or agents.

## §Integrations
### §Model Providers

Rig natively supports the following completion and embedding model provider integrations:

- OpenAI
- Cohere
- Anthropic
- Perplexity
- Gemini

You can also implement your own model provider integration by defining types that implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits.

### §Vector Stores

Rig currently supports the following vector store integrations via companion crates:

- `rig-mongodb`: Vector store implementation for MongoDB
- `rig-lancedb`: Vector store implementation for LanceDB
- `rig-neo4j`: Vector store implementation for Neo4j
- `rig-qdrant`: Vector store implementation for Qdrant

You can also implement your own vector store integration by defining types that implement the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait.

- `pub use embeddings::[Embed](https://docs.rs/rig-core/latest/rig/embeddings/embed/trait.Embed.html "trait rig::embeddings::embed::Embed");`

- This module contains the implementation of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") struct and its builder.

- This module provides functionality for working with completion models. It provides traits, structs, and enums for generating completion requests, handling completion responses, and defining completion models.
- This module provides functionality for working with embeddings. Embeddings are numerical representations of documents or other objects, typically used in natural language processing (NLP) tasks such as text classification, information retrieval, and document similarity.
- This module provides high-level abstractions for extracting structured data from text using LLMs.
- This module provides utility structs for loading and preprocessing files.

- This module defines a flexible pipeline API for defining a sequence of operations that may or may not use AI components (e.g.: semantic search, LLMs prompting, etc).
- This module contains clients for the different LLM providers that Rig supports.
- Module defining tool related structs and traits.---
title: "rig - Rust"
source: "https://docs.rs/rig-core/latest/rig/#model-providers"
author:
published:
created: 2024-12-23
description: "Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity."
tags:
  - "clippings"
---
Expand description

Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.

## §Table of contents

- [High-level features](https://docs.rs/rig-core/latest/rig/#high-level-features)
- [Simple Example](https://docs.rs/rig-core/latest/rig/#simple-example)
- [Core Concepts](https://docs.rs/rig-core/latest/rig/#core-concepts)
- [Integrations](https://docs.rs/rig-core/latest/rig/#integrations)

## §High-level features

- Full support for LLM completion and embedding workflows
- Simple but powerful common abstractions over LLM providers (e.g. OpenAI, Cohere) and vector stores (e.g. MongoDB, in-memory)
- Integrate LLMs in your app with minimal boilerplate

## §Simple example:

```
use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and agent.
    // This requires the \`OPENAI_API_KEY\` environment variable to be set.
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.agent("gpt-4").build();

    // Prompt the model and print its response
    let response = gpt4
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
```

Note: using `#[tokio::main]` requires you enable tokio’s `macros` and `rt-multi-thread` features or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

## §Core concepts
### §Completion and embedding models

Rig provides a consistent API for working with LLMs and embeddings. Specifically, each provider (e.g. OpenAI, Cohere) has a `Client` struct that can be used to initialize completion and embedding models. These models implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits respectively, which provide a common, low-level interface for creating completion and embedding requests and executing them.

### §Agents

Rig also provides high-level abstractions over LLMs in the form of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type.

The [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type can be used to create anything from simple agents that use vanilla models to full blown RAG systems that can be used to answer questions using a knowledge base.

### §Vector stores and indexes

Rig provides a common interface for working with vector stores and indexes. Specifically, the library provides the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait, which can be implemented to define vector stores and indices respectively. Those can then be used as the knowledge base for a RAG enabled [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent"), or as a source of context documents in a custom architecture that use multiple LLMs or agents.

## §Integrations
### §Model Providers

Rig natively supports the following completion and embedding model provider integrations:

- OpenAI
- Cohere
- Anthropic
- Perplexity
- Gemini

You can also implement your own model provider integration by defining types that implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits.

### §Vector Stores

Rig currently supports the following vector store integrations via companion crates:

- `rig-mongodb`: Vector store implementation for MongoDB
- `rig-lancedb`: Vector store implementation for LanceDB
- `rig-neo4j`: Vector store implementation for Neo4j
- `rig-qdrant`: Vector store implementation for Qdrant

You can also implement your own vector store integration by defining types that implement the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait.

- `pub use embeddings::[Embed](https://docs.rs/rig-core/latest/rig/embeddings/embed/trait.Embed.html "trait rig::embeddings::embed::Embed");`

- This module contains the implementation of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") struct and its builder.

- This module provides functionality for working with completion models. It provides traits, structs, and enums for generating completion requests, handling completion responses, and defining completion models.
- This module provides functionality for working with embeddings. Embeddings are numerical representations of documents or other objects, typically used in natural language processing (NLP) tasks such as text classification, information retrieval, and document similarity.
- This module provides high-level abstractions for extracting structured data from text using LLMs.
- This module provides utility structs for loading and preprocessing files.

- This module defines a flexible pipeline API for defining a sequence of operations that may or may not use AI components (e.g.: semantic search, LLMs prompting, etc).
- This module contains clients for the different LLM providers that Rig supports.
- Module defining tool related structs and traits.---
title: "rig - Rust"
source: "https://docs.rs/rig-core/latest/rig/"
author:
published:
created: 2024-12-23
description: "Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity."
tags:
  - "clippings"
---
Expand description

Rig is a Rust library for building LLM-powered applications that focuses on ergonomics and modularity.

## §Table of contents

- [High-level features](https://docs.rs/rig-core/latest/rig/#high-level-features)
- [Simple Example](https://docs.rs/rig-core/latest/rig/#simple-example)
- [Core Concepts](https://docs.rs/rig-core/latest/rig/#core-concepts)
- [Integrations](https://docs.rs/rig-core/latest/rig/#integrations)

## §High-level features

- Full support for LLM completion and embedding workflows
- Simple but powerful common abstractions over LLM providers (e.g. OpenAI, Cohere) and vector stores (e.g. MongoDB, in-memory)
- Integrate LLMs in your app with minimal boilerplate

## §Simple example:

```
use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and agent.
    // This requires the \`OPENAI_API_KEY\` environment variable to be set.
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.agent("gpt-4").build();

    // Prompt the model and print its response
    let response = gpt4
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
```

Note: using `#[tokio::main]` requires you enable tokio’s `macros` and `rt-multi-thread` features or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

## §Core concepts
### §Completion and embedding models

Rig provides a consistent API for working with LLMs and embeddings. Specifically, each provider (e.g. OpenAI, Cohere) has a `Client` struct that can be used to initialize completion and embedding models. These models implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits respectively, which provide a common, low-level interface for creating completion and embedding requests and executing them.

### §Agents

Rig also provides high-level abstractions over LLMs in the form of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type.

The [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") type can be used to create anything from simple agents that use vanilla models to full blown RAG systems that can be used to answer questions using a knowledge base.

### §Vector stores and indexes

Rig provides a common interface for working with vector stores and indexes. Specifically, the library provides the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait, which can be implemented to define vector stores and indices respectively. Those can then be used as the knowledge base for a RAG enabled [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent"), or as a source of context documents in a custom architecture that use multiple LLMs or agents.

## §Integrations
### §Model Providers

Rig natively supports the following completion and embedding model provider integrations:

- OpenAI
- Cohere
- Anthropic
- Perplexity
- Gemini

You can also implement your own model provider integration by defining types that implement the [CompletionModel](https://docs.rs/rig-core/latest/rig/completion/trait.CompletionModel.html "trait rig::completion::CompletionModel") and [EmbeddingModel](https://docs.rs/rig-core/latest/rig/embeddings/embedding/trait.EmbeddingModel.html "trait rig::embeddings::embedding::EmbeddingModel") traits.

### §Vector Stores

Rig currently supports the following vector store integrations via companion crates:

- `rig-mongodb`: Vector store implementation for MongoDB
- `rig-lancedb`: Vector store implementation for LanceDB
- `rig-neo4j`: Vector store implementation for Neo4j
- `rig-qdrant`: Vector store implementation for Qdrant

You can also implement your own vector store integration by defining types that implement the [VectorStoreIndex](https://docs.rs/rig-core/latest/rig/vector_store/trait.VectorStoreIndex.html "trait rig::vector_store::VectorStoreIndex") trait.

- `pub use embeddings::[Embed](https://docs.rs/rig-core/latest/rig/embeddings/embed/trait.Embed.html "trait rig::embeddings::embed::Embed");`

- This module contains the implementation of the [Agent](https://docs.rs/rig-core/latest/rig/agent/struct.Agent.html "struct rig::agent::Agent") struct and its builder.

- This module provides functionality for working with completion models. It provides traits, structs, and enums for generating completion requests, handling completion responses, and defining completion models.
- This module provides functionality for working with embeddings. Embeddings are numerical representations of documents or other objects, typically used in natural language processing (NLP) tasks such as text classification, information retrieval, and document similarity.
- This module provides high-level abstractions for extracting structured data from text using LLMs.
- This module provides utility structs for loading and preprocessing files.

- This module defines a flexible pipeline API for defining a sequence of operations that may or may not use AI components (e.g.: semantic search, LLMs prompting, etc).
- This module contains clients for the different LLM providers that Rig supports.
- Module defining tool related structs and traits.---
title: "rig/README.md at main · 0xPlaygrounds/rig"
source: "https://github.com/0xPlaygrounds/rig/blob/main/README.md"
author:
  - "[[GitHub]]"
published:
created: 2024-12-23
description: "⚙️🦀 Build portable, modular & lightweight Fullstack Agents - rig/README.md at main · 0xPlaygrounds/rig"
tags:
  - "clippings"
---
[Open in github.dev](https://github.dev/) [Open in a new github.dev tab](https://github.dev/) [Open in codespace](https://github.com/codespaces/new/0xPlaygrounds/rig/tree/main?resume=1)

  ## Files

## Latest commit

  ![Rig logo](https://github.com/0xPlaygrounds/rig/raw/main/img/rig-playgrounds-light.svg)  
[![](https://camo.githubusercontent.com/cb4bfad8d48a059878292bc9ce6d709d21352e96689bf27c345e9b7c81fda54d/68747470733a2f2f696d672e736869656c64732e696f2f62616467652ff09f939620646f63732d7269672e72732d6463613238322e737667)](https://docs.rig.rs/)   [![](https://camo.githubusercontent.com/6ad6b41a20a20a20ae3da1fa622c591edeaf87305c44f8d9401e3c8a862090a6/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f646f63732d415049205265666572656e63652d6463613238322e737667)](https://docs.rs/rig-core/latest/rig/)   [![](https://camo.githubusercontent.com/8a678cb06fa9b5f34cd9dbbcfda398d7063a13904e1b0d60c5f12d7749680257/68747470733a2f2f696d672e736869656c64732e696f2f6372617465732f762f7269672d636f72652e7376673f636f6c6f723d646361323832)](https://crates.io/crates/rig-core)   [![](https://camo.githubusercontent.com/a037b33097514c3a0314e0b4cb028767fb545aa9a4e0eae1b553825ae7d4a24d/68747470733a2f2f696d672e736869656c64732e696f2f6372617465732f642f7269672d636f72652e7376673f636f6c6f723d646361323832)](https://crates.io/crates/rig-core)  
[![](https://camo.githubusercontent.com/317fd05e6a85c858bcb7ec5cfd81a718f67fce8cf26e75d812b83ddc923797e2/68747470733a2f2f696d672e736869656c64732e696f2f646973636f72642f3531313330333634383131393232363338323f636f6c6f723d253233366438326363266c6162656c3d446973636f7264266c6f676f3d646973636f7264266c6f676f436f6c6f723d7768697465)](https://discord.gg/playgrounds)   [![stars - rig](https://camo.githubusercontent.com/e4a634af747694261526d0b6ac5a8095a7aceef1014664a44e6ca7b8bce25c66/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f3078506c617967726f756e64732f7269673f7374796c653d736f6369616c)](https://github.com/0xPlaygrounds/rig)  
[![](https://camo.githubusercontent.com/285fac7e67356ca626e865e96ff7c420cfb97d00ac3d113231916615516ab7b2/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f6275696c745f776974682d527573742d6463613238322e7376673f6c6f676f3d72757374)](https://github.com/0xPlaygrounds/rig/blob/main)   [![](https://camo.githubusercontent.com/5c6e23699339e4de5dfb09f54e2a309aefcd5d5e1bd9629a01695b73c033f11d/68747470733a2f2f696d672e736869656c64732e696f2f747769747465722f666f6c6c6f772f506c617967726f756e64733078)](https://twitter.com/Playgrounds0x)    

 

✨ If you would like to help spread the word about Rig, please consider starring the repo!

Warning

Here be dragons! As we plan to ship a torrent of features in the following months, future updates **will** contain **breaking changes**. With Rig evolving, we'll annotate changes and highlight migration paths as we encounter them.

## What is Rig?

Rig is a Rust library for building scalable, modular, and ergonomic **LLM-powered** applications.

More information about this crate can be found in the [official](https://docs.rig.rs/) & [crate](https://docs.rs/rig-core/latest/rig/) (API Reference) documentations.

Help us improve Rig by contributing to our [Feedback form](https://bit.ly/Rig-Feeback-Form).

## Table of contents

- [What is Rig?](https://github.com/0xPlaygrounds/rig/blob/main/#what-is-rig)
- [Table of contents](https://github.com/0xPlaygrounds/rig/blob/main/#table-of-contents)
- [High-level features](https://github.com/0xPlaygrounds/rig/blob/main/#high-level-features)
- [Get Started](https://github.com/0xPlaygrounds/rig/blob/main/#get-started)
- [Simple example:](https://github.com/0xPlaygrounds/rig/blob/main/#simple-example)
- [Integrations](https://github.com/0xPlaygrounds/rig/blob/main/#integrations)

## High-level features

- Full support for LLM completion and embedding workflows
- Simple but powerful common abstractions over LLM providers (e.g. OpenAI, Cohere) and vector stores (e.g. MongoDB, in-memory)
- Integrate LLMs in your app with minimal boilerplate

## Get Started

### Simple example:

```
use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and model
    // This requires the \`OPENAI_API_KEY\` environment variable to be set.
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.agent("gpt-4").build();

    // Prompt the model and print its response
    let response = gpt4
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
```

Note using `#[tokio::main]` requires you enable tokio's `macros` and `rt-multi-thread` features or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

You can find more examples each crate's `examples` (ie. [`rig-core/examples`](https://github.com/0xPlaygrounds/rig/blob/main/rig-core/examples)) directory. More detailed use cases walkthroughs are regularly published on our [Dev.to Blog](https://dev.to/0thtachi) and added to Rig's official documentation [(docs.rig.rs)](http://docs.rig.rs/).

## Supported Integrations

| Model Providers | Vector Stores |
| --- | --- |
| [![ChatGPT logo](https://camo.githubusercontent.com/cef5f80108da8639b2205539f47968976862fb1e04d2fed830771bc756313456/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f7468756d622f302f30342f436861744750545f6c6f676f2e7376672f3130323470782d436861744750545f6c6f676f2e7376672e706e67)](https://camo.githubusercontent.com/cef5f80108da8639b2205539f47968976862fb1e04d2fed830771bc756313456/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f7468756d622f302f30342f436861744750545f6c6f676f2e7376672f3130323470782d436861744750545f6c6f676f2e7376672e706e67) ![Claude Anthropic logo](https://camo.githubusercontent.com/7311b190daae505797b4084c5677027ac4c5f82039124389f1c53f2f2c2fc58e/68747470733a2f2f7777772e666168696d61692e636f6d2f77702d636f6e74656e742f75706c6f6164732f323032342f30362f556e7469746c65642d64657369676e2d372e706e67)   [![Cohere logo](https://camo.githubusercontent.com/6ebe007f25b08060fbb8442c6630daaa2e48ee0121a2242f41e953cae0d26b1e/68747470733a2f2f63646e2e73616e6974792e696f2f696d616765732f726a74716d7766752f70726f64756374696f6e2f306164626633393434333966346364306162386235623362366665316461313063383039393032342d323031783230302e737667)](https://camo.githubusercontent.com/6ebe007f25b08060fbb8442c6630daaa2e48ee0121a2242f41e953cae0d26b1e/68747470733a2f2f63646e2e73616e6974792e696f2f696d616765732f726a74716d7766752f70726f64756374696f6e2f306164626633393434333966346364306162386235623362366665316461313063383039393032342d323031783230302e737667) [![Gemini logo](https://camo.githubusercontent.com/2d6020bd2bd376424c003989ea2ec63ef6d293ff8b7df8cb5efc7f6b526fc749/68747470733a2f2f6c6f676f73706e672e6f72672f646f776e6c6f61642f676f6f676c652d67656d696e692f676f6f676c652d67656d696e692d313032342e706e67)](https://camo.githubusercontent.com/2d6020bd2bd376424c003989ea2ec63ef6d293ff8b7df8cb5efc7f6b526fc749/68747470733a2f2f6c6f676f73706e672e6f72672f646f776e6c6f61642f676f6f676c652d67656d696e692f676f6f676c652d67656d696e692d313032342e706e67)   [![xAI logo](https://camo.githubusercontent.com/00a9dcc912dda396c8a03b71d4fabfd4d7e50524e161efb105200b1d237c8167/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f7468756d622f352f35372f5841492d4c6f676f2e7376672f35313270782d5841492d4c6f676f2e7376672e706e673f3230323430393132323232383431)](https://camo.githubusercontent.com/00a9dcc912dda396c8a03b71d4fabfd4d7e50524e161efb105200b1d237c8167/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f7468756d622f352f35372f5841492d4c6f676f2e7376672f35313270782d5841492d4c6f676f2e7376672e706e673f3230323430393132323232383431) [![perplexity logo](https://private-user-images.githubusercontent.com/26660420/386746427-4763ae96-ddc9-4f69-ab38-23592e6c4ead.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzQ5NjMyMjgsIm5iZiI6MTczNDk2MjkyOCwicGF0aCI6Ii8yNjY2MDQyMC8zODY3NDY0MjctNDc2M2FlOTYtZGRjOS00ZjY5LWFiMzgtMjM1OTJlNmM0ZWFkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjIzVDE0MDg0OFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM4ZmJmMmMxMzQ4NTJhZDYyMmM1ZTExNDNhYjI4ZmQ1NGUwMWI1ODAwYTQyZGJlZDJlMjIwZmFjMzE5ZmFmZDgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.vjCvO7h1BKWAlL2AizJVpuKls6RIwhywAt7ZJ4c3gyE)](https://private-user-images.githubusercontent.com/26660420/386746427-4763ae96-ddc9-4f69-ab38-23592e6c4ead.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzQ5NjMyMjgsIm5iZiI6MTczNDk2MjkyOCwicGF0aCI6Ii8yNjY2MDQyMC8zODY3NDY0MjctNDc2M2FlOTYtZGRjOS00ZjY5LWFiMzgtMjM1OTJlNmM0ZWFkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEyMjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMjIzVDE0MDg0OFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWM4ZmJmMmMxMzQ4NTJhZDYyMmM1ZTExNDNhYjI4ZmQ1NGUwMWI1ODAwYTQyZGJlZDJlMjIwZmFjMzE5ZmFmZDgmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.vjCvO7h1BKWAlL2AizJVpuKls6RIwhywAt7ZJ4c3gyE) | [![Mongo DB logo](https://camo.githubusercontent.com/7563877c2bdce93d15a63116b8b942a6b7b18d7c5194e530d2bf13912a6e5e32/68747470733a2f2f63646e2e70726f642e776562736974652d66696c65732e636f6d2f3636343063643238663531663133313735653537376330352f3636346530306134303065323366313034656432623663645f33623364643665382d386137332d353837392d383461392d6134326435623931306337342e737667)](https://camo.githubusercontent.com/7563877c2bdce93d15a63116b8b942a6b7b18d7c5194e530d2bf13912a6e5e32/68747470733a2f2f63646e2e70726f642e776562736974652d66696c65732e636f6d2f3636343063643238663531663133313735653537376330352f3636346530306134303065323366313034656432623663645f33623364643665382d386137332d353837392d383461392d6134326435623931306337342e737667) [![Neo4j logo](https://camo.githubusercontent.com/eb5499b6ec38bd7c43366b92e6cb389b8d3581d1c17c10fc33ecfc1090be5805/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f652f65352f4e656f346a2d6c6f676f5f636f6c6f722e706e67)](https://camo.githubusercontent.com/eb5499b6ec38bd7c43366b92e6cb389b8d3581d1c17c10fc33ecfc1090be5805/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f652f65352f4e656f346a2d6c6f676f5f636f6c6f722e706e67)  [![Lance DB logo](https://camo.githubusercontent.com/60f17ad265034b515600353da3a2b7ba9afbad155c2fa085b9890bb1c472fa72/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f3834342f312a4a7036567746304f63646579527957304c6e30524d514032782e706e67)](https://camo.githubusercontent.com/60f17ad265034b515600353da3a2b7ba9afbad155c2fa085b9890bb1c472fa72/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f3834342f312a4a7036567746304f63646579527957304c6e30524d514032782e706e67) |    

Vector stores are available as separate companion-crates:

- MongoDB vector store: [`rig-mongodb`](https://github.com/0xPlaygrounds/rig/tree/main/rig-mongodb)
- LanceDB vector store: [`rig-lancedb`](https://github.com/0xPlaygrounds/rig/tree/main/rig-lancedb)
- Neo4j vector store: [`rig-neo4j`](https://github.com/0xPlaygrounds/rig/tree/main/rig-neo4j)
- Qdrant vector store: [`rig-qdrant`](https://github.com/0xPlaygrounds/rig/tree/main/rig-qdrant)

[![Build by Playgrounds](https://github.com/0xPlaygrounds/rig/raw/main/img/built-by-playgrounds.svg)](https://github.com/0xPlaygrounds/rig/blob/main/img/built-by-playgrounds.svg)---
title: "rig/rig-neo4j at main · 0xPlaygrounds/rig"
source: "https://github.com/0xPlaygrounds/rig/tree/main/rig-neo4j"
author:
published:
created: 2024-12-23
description:
tags:
  - "clippings"
---
![Rig logo](https://github.com/0xPlaygrounds/rig/raw/main/img/rig_logo.svg)

+

  ![Neo4j logo](https://camo.githubusercontent.com/d44f80539e57e4b677962dc4a30a842d3c37dc1ecbedc7f7a882aa78644fbc23/68747470733a2f2f636f6d6d6f6e732e77696b696d656469612e6f72672f77696b692f46696c653a4e656f346a2d6c6f676f5f636f6c6f722e706e67)

This companion crate implements a Rig vector store based on Neo4j Graph database. It uses the [neo4rs](https://github.com/neo4j-labs/neo4rs) crate to interact with Neo4j. Note that the neo4rs crate is a work in progress and does not yet support all Neo4j features. Further documentation on Neo4j & vector search integration can be found on the [neo4rs docs](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/).

## Prerequisites

The GenAI plugin is enabled by default in Neo4j Aura.

The plugin needs to be installed on self-managed instances. This is done by moving the neo4j-genai.jar file from /products to /plugins in the Neo4j home directory, or, if you are using Docker, by starting the Docker container with the extra parameter --env NEO4J\_PLUGINS='\["genai"\]'. For more information, see Operations Manual → Configure plugins.

## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```
[dependencies]
rig-neo4j = "0.1"
```

You can also run `cargo add rig-neo4j rig-core` to add the most recent versions of the dependencies to your project.

See the [examples](https://github.com/0xPlaygrounds/rig/blob/main/rig-neo4j/examples) folder for usage examples.

- [examples/vector\_search\_simple.rs](https://github.com/0xPlaygrounds/rig/blob/main/rig-neo4j/examples/vector_search_simple.rs) shows how to create an index on simple data.
- [examples/vector\_search\_movies\_consume.rs](https://github.com/0xPlaygrounds/rig/blob/main/rig-neo4j/examples/vector_search_movies_consume.rs) shows how to query an existing index.
- [examples/vector\_search\_movies\_create.rs](https://github.com/0xPlaygrounds/rig/blob/main/rig-neo4j/examples/vector_search_movies_create.rs) shows how to create embeddings & index on a large DB and query it in one go.

## Notes

- The `rig-neo4j::vector_index` module offers utility functions to create and query a Neo4j vector index. You can also create indexes using the Neo4j browser or directly call cypther queries with the Neo4rs crate. See the [Neo4j documentation](https://neo4j.com/docs/genai/tutorials/embeddings-vector-indexes/setup/vector-index/) for more information. Example [examples/vector\_search\_simple.rs](https://github.com/0xPlaygrounds/rig/blob/main/rig-neo4j/examples/vector_search_simple.rs) shows how to create an index on existing data.

```
CREATE VECTOR INDEX moviePlots
FOR (m:Movie)
ON m.embedding
OPTIONS {indexConfig: {
    \`vector.dimensions\`: 1536,
    \`vector.similarity_function\`: 'cosine'
}}
```

## Roadmap

- Add support for creating the vector index through RIG.
- Add support for adding embeddings to an existing database
- Add support for uploading documents to an existing database---
title: "⚡🦀 Deploy a blazing-fast & Lightweight LLM app with Rust-Rig-LanceDB"
source: "https://dev.to/garance_buricatu_a6864136/deploy-a-blazing-fast-lightweight-llm-app-with-rust-rig-lancedb-139l"
author:
  - "[[Garance Buricatu]]"
published: 2024-11-22
created: 2024-12-23
description: "TL;DR    A step-by-step walkthrough on deploying a LLM app using Rig &amp; LanceDB on AWS... Tagged with rust, ai, devops, aws."
tags:
  - "clippings"
---
## TL;DR

- A step-by-step walkthrough on deploying a LLM app using [`Rig`](https://github.com/0xPlaygrounds/rig) & [`LanceDB`](https://lancedb.com/) on AWS Lambda. You'll learn how to prepare your app, choose the right storage backend (like S3 or EFS), and optimize performance by efficiently using cloud metrics.
- **Stats: Rig RAG Agent using LanceDB on AWS :**
1. Low memory usage (96MB - 113MB)
2. Fast cold starts (consistently 160ms)
- **Stats: LangChain RAG Agent using LanceDB on AWS:**
1. Higher memory usage (246MB - 360MB)
2. Slower cold starts (1,900ms - 2,700ms)
- [**Jump to Metrics** ⏬](https://dev.to/garance_buricatu_a6864136/#final-comparison-between-rig-and-langchain)

## Table of Contents

- [How to Deploy Your Rig App with LanceDB: A Step-by-Step Guide](https://dev.to/garance_buricatu_a6864136/#how-to-deploy-your-rig-app-with-lancedb-a-step-by-step-guide)
- [Introduction](https://dev.to/garance_buricatu_a6864136/#introduction)
- [Prerequisites](https://dev.to/garance_buricatu_a6864136/#prerequisites)
- [Our use case: Montreal 🌇](https://dev.to/garance_buricatu_a6864136/#our-use-case-montreal-)
- [LanceDB Quick Overview 💾](https://dev.to/garance_buricatu_a6864136/#lancedb-quick-overview-)
- [LanceDB Storage Backends](https://dev.to/garance_buricatu_a6864136/#lancedb-storage-backends)
- [S3 - Object Store](https://dev.to/garance_buricatu_a6864136/#s3---object-store)
- [Lambda ephemeral storage - Local file system](https://dev.to/garance_buricatu_a6864136/#lambda-ephemeral-storage---local-file-system)
- [EFS - Virtual file system](https://dev.to/garance_buricatu_a6864136/#efs---virtual-file-system)
- [Metrics on the cloud ☁️](https://dev.to/garance_buricatu_a6864136/#metrics-on-the-cloud-)
- [Memory, CPU, and runtime](https://dev.to/garance_buricatu_a6864136/#memory-cpu-and-runtime)
- [Langchain Montreal Agent App 🐍](https://dev.to/garance_buricatu_a6864136/#langchain-montreal-agent-app-)
- [Deployment package](https://dev.to/garance_buricatu_a6864136/#deployment-package-1)
- [Memory, CPU, and runtime](https://dev.to/garance_buricatu_a6864136/#memory-cpu-and-runtime-1)
- [Final Comparison between Rig and LangChain](https://dev.to/garance_buricatu_a6864136/#final-comparison-between-rig-and-langchain)
- [Resources](https://dev.to/garance_buricatu_a6864136/#resources)

## Introduction

Welcome back to **Deploy Your Rig Application**! Apps built with Rig vary in complexity based on LLM usage, vector databases for RAG, and infrastructure deployment. This series explores various configurations for production use.

⭐ **Today's Highlight**: Rig's **LanceDB integration**! ⭐

We'll deploy a Rig agent using OpenAI's `text-embedding-ada-002` and `GPT-4o`, relying on the [LanceDB vector store](https://lancedb.com/) and deployed on [AWS Lambda](https://aws.amazon.com/lambda/).

> *💡 If you're new to Rig and want to start from the beginning or are looking for additional tutorials, check out our [blog series](https://rig.rs/build-with-rig-guide.html).*

Let’s dive in!

## Prerequisites

Before we begin building, ensure you have the following:

> ❗ We will *not* be covering how to write your RAG app with Rig, only how to deploy it. So make sure you read [this tutorial](https://dev.to/0thtachi/build-a-fast-and-lightweight-rust-vector-search-app-with-rig-lancedb-57h2) first to help you code your application.
> 
> - A clone of the [`rig-montreal-lancedb`](https://github.com/garance-buricatu/rig-aws/tree/master/rig-montreal-lancedb) crate which includes two separate binaries: a [`loader`](https://github.com/garance-buricatu/rig-aws/blob/master/rig-montreal-lancedb/src/bin/loader.rs) (writes data to LanceDB) and an [`app`](https://github.com/garance-buricatu/rig-aws/blob/master/rig-montreal-lancedb/src/bin/app.rs) (performs RAG on LanceDB).
> - An AWS account and some background knowledge on deployments on AWS, including Cloudformation templates
> - An OpenAI api key

## Our use case: Montreal 🌇

The app in [`rig-montreal-lancedb`](https://github.com/garance-buricatu/rig-aws/tree/master/rig-montreal-lancedb) RAGs data from [montreal open data](https://donnees.montreal.ca/). The Montréal municipality generates and manages large quantities of data through its activities, such as data about agriculture, politics, transportation, health and much more. The open data app publishes all these datasets and make them freely accessible to all citizens! Our app will index the metadata of all the public datasets so that a user can ask questions pertaining to the open data.  
The `loader` binary indexes all dataset metadata (name, description, tags, ...) into LanceDB and the `app` binary performs vector search on the data based on a prompt. For example:

> **Prompt:** Give me information on gaseous pollutants in Montreal. How are the concentrations measured?  
> **App answer:** The concentrations of gaseous pollutants in Montreal are measured through the Réseau de surveillance de la qualité de l'air (RSQA), which is a network of measurement stations located on the Island of Montreal. These stations continuously determine the atmospheric concentration of various pollutants. The data is transmitted via telemetry, ...

## LanceDB Quick Overview 💾

[**Lance**](https://github.com/lancedb/lance) is an **open-source columnar data format** designed for performant ML workloads.

- Written in Rust 🦀.
- Native support for storing, querying and filtering vectors, deeply nested data and multi-modal data (text, images, videos, point clouds, and more).
- Support for vector similarity search, full-text search and SQL.
- Interoperable with other columnar formats (such as Parquet) via [Arrow](https://arrow.apache.org/overview/)
- Disk-based indexes and storage.
- Built to scale to hundreds of terabytes of data.

[**LanceDB**](https://lancedb.github.io/lancedb/) is an **open-source vector database**.

- Written in Rust 🦀.
- Built on top of Lance.
- Support for Python, JavaScript, and Rust client libraries to interact with the database.
- Allows storage of raw data, metadata, and embeddings all at once.

## LanceDB Storage Options

LanceDB's underlying optimized storage format, `lance`, is flexible enough to be supported by various storage backends, such as local NVMe, [EBS](https://aws.amazon.com/ebs/), [EFS](https://aws.amazon.com/efs/), [S3](https://aws.amazon.com/s3/) and other third-party APIs that connect to the cloud.

> 💡 All you need to do to use a specific storage backend is define its connection string in the LanceDB client!

Let's go through some storage options that are compatible with AWS Lambda!

## S3 - Object Store

> ❕Data is stored as individual objects *all at the same level*.  
> ❕Objects are kept track of by a distributed hash table (DHT), where each object is identified by a unique ID.

| **Pros of Object Stores** | **Cons of Object Stores** |
| --- | --- |
| **Unlimited scaling** ♾️: Objects can be stored across distributed systems, eliminating single-node limitations. This is ideal for ML and AI applications handling large data volumes. | **Higher latency** 🚚: Accessing a remote object store over a network via HTTP/HTTPS adds overhead compared to file system protocols like NFS. Additionally, storing metadata separately from objects introduces some retrieval latency. |
| **Cheap** 💸: The simple storage design makes it more affordable than traditional file systems. |  |
| **Highly available** and **resilient** 💪: Affordable storage allows for redundant data storage within and across data centers. |  |

### S3 + LanceDB setup on AWS lambda

**⚠️ Important**: S3 does **not support concurrent writes**. If multiple processes attempt to write to the same table simultaneously, it could lead to data corruption. But there's a solution! Use the [DynamoDB commit store feature in LanceDB](https://lancedb.github.io/lancedb/guides/storage/#dynamodb-commit-store-for-concurrent-writes) to prevent this.

---

#### Part I - Write lambda function code

1. Create an [S3 Bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html) where your Lance database will be stored. Ours is called: `rig-montreal-lancedb`.
2. In the lambda code, connect to the store via the [`LanceBD client`](https://docs.rs/lancedb/latest/lancedb/connection/struct.Connection.html) as so:

```
// Note: Create s3://rig-montreal-lancedb bucket beforehand
let db = lancedb::connect("s3://rig-montreal-lancedb").execute().await?;
// OR
let db = lancedb::connect("s3+ddb://rig-montreal-lancedb?ddbTableName=my-dynamodb-table").execute().await?;
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

#### Part II - Deploy lambdas

> 💡 Need a refresher on Lambda deployments? Check out our [previous blog](https://dev.to/garance_buricatu_a6864136/how-to-deploy-your-rig-app-on-aws-lambda-a-step-by-step-guide-2ge5) for a full walkthrough.  

```
# Lambda that writes to the store
cargo lambda build --release --bin loader
cargo lambda deploy --binary-name loader <your_loader_function_name>

# Lambda that reads to the store
cargo lambda build --release --bin app
cargo lambda deploy --binary-name app <your_app_function_name>
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

> 💡 Don’t forget to set the necessary [IAM permissions](https://lancedb.github.io/lancedb/guides/storage/#aws-iam-permissions)! Your lambda functions need appropriate access to the S3 bucket — whether it’s read, write, or both.

## Lambda ephemeral storage - Local file system

[Lambda ephemeral storage](https://docs.aws.amazon.com/lambda/latest/dg/configuration-ephemeral-storage.html) is **temporary and unique** to each execution environment, it is not intended for persistent storage. In other words, any LanceDB store created during the lambda execution on ephemeral storage will be wiped when the function cold starts.  
This option can be used for very specific use cases (mostly for testing) where writing to the store needs to be done in the same process as reading, and data is only read by a single lambda execution.

Ephemeral storage in a lambda is found in the `/tmp` directory. All you need to do is:  

```
let db = lancedb::connect("/tmp").execute().await?;
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

## EFS - Virtual file system

> ❕A **serverless**, **elastic**, **shared file system** designed to be consumed by AWS services like EC2 and Lambda.  
> ❕Data is **persisted** and can be shared across lambda invocations (unlike the S3 without commit store and ephemeral storage options above).  
> ❕Supports up to 25,000 **concurrent connections**.

| **Pros of EFS** | **Cons of EFS** |
| --- | --- |
| **Stateful lambda**: Mounting an EFS instance on a lambda function provides knowledge of previous and concurrent executions. | **Development time**: More involved cloud setup |
| **Low latency** ⚡: A lambda function resides in the same **VPC** as the EFS instance, allowing low-latency network calls via the **NFS** protocol. | **Cost** 💲: More expensive than S3 |

### EFS + LanceDB setup on AWS Lambda

> 💡 Setting up EFS in the cloud can be intricate, so you can use our [CloudFormation template](https://github.com/garance-buricatu/rig-aws/blob/master/rig-montreal-lancedb/template.yaml) to streamline the deployment process.

#### Part I - Build Rust code and upload zip files to S3

In the lambda code, connect to the store via the [LanceBD client](https://docs.rs/lancedb/latest/lancedb/connection/struct.Connection.html) as so:  

```
let db = lancedb::connect("/mnt/efs").execute().await?;
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

Then, compile your code, zip the binaries, and upload them to S3:  

```
# Can also do this directly on the AWS console
aws s3api create-bucket --bucket <your_bucket_name>

cargo lambda build --release --bin loader
cargo lambda build --release --bin app

cd target/lambda/loader
zip -r bootstrap.zip bootstrap
# Can also do this directly on the AWS console
aws s3 cp bootstrap.zip s3://<your_bucket_name>/rig/loader/

cd ..
zip -r bootstrap.zip bootstrap
# Can also do this directly on the AWS console
aws s3 cp bootstrap.zip s3://<your_bucket_name>/rig/app/
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

#### Part II - Understand Cloudformation template

The [template](https://github.com/garance-buricatu/rig-aws/blob/master/rig-montreal-lancedb/template.yaml) assumes that your AWS account already has the following resources:

1. A **VPC** with at least two private subnets in separate availability zones, each with public internet access.
2. An **S3 bucket** (as created in Part I) for storing Lambda code. > 💡 If you’re missing these resources, follow this AWS [tutorial](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-example-private-subnets-nat.html) to set up a basic VPC and subnets.

**EFS setup**

1. **Mount Targets:** Create two mount targets for your EFS instance — one in each subnet (specified in `Parameters` section of CFT template).
2. **Security Groups:** Set up an EFS security group with rules to allow **NFS traffic** from your Lambda functions’ security group.

**Lambda functions setup**

1. **Loader and App Lambdas:** Deploy both Lambda functions (`loader` and `app`) in the same subnets as your EFS mount targets.
2. **Security Groups:** Assign a security group that enables access to the EFS security group and public internet.
3. **EFS Mounting:** Configure the Lambdas to mount the EFS targets at `/mnt/efs`.

> ## 💡 Once everything’s ready, deploy the CloudFormation template to launch your environment with just one click!

### Metrics on the cloud ☁️

If you've made it to here, you have the Montreal rig app with EFS as the LanceDbB storage backend deployed on AWS Lambda! 🎉 Now we want to look at some metrics when the app is run in the cloud.

For reference, we replicated the Montreal agent using [langchain 🐍](https://python.langchain.com/) in this [python project](https://github.com/garance-buricatu/rig-aws/tree/master/langchain-montreal-lancedb) which contains the source code for the [`loader`](https://github.com/garance-buricatu/rig-aws/blob/master/langchain-montreal-lancedb/loader.py) and [`app`](https://github.com/garance-buricatu/rig-aws/blob/master/langchain-montreal-lancedb/app.py) lambdas. The python app uses the same LanceDB vector store on the same EFS instance as the Rig app. To see how the python app was configured in the cloud, take a look at the [CloudFormation template](https://github.com/garance-buricatu/rig-aws/blob/master/rig-montreal-lancedb/template.yaml).

Let's compare them!

### Rig - Memory, runtime, and coldstarts

We invoked the `app` function 50 times for each memory configuration of 128MB, 256MB, 512MB, 1024MB using the [power tuner tool](https://github.com/alexcasalboni/aws-lambda-power-tuning).  
The Cloudwatch query below gathers averages about runtime, memory usage, and cold starts of the lambda over the 50 invocations.  

```
filter @type = "REPORT"
| stats 
      avg(@maxMemoryUsed) / 1000000 as MemoryUsageMB,
      avg(@duration) / 1000 as AvgDurationSec,
      max(@duration) / 1000 as MaxDurationSec, 
      min(@duration) / 1000 as MinDurationSec, 
      avg(@initDuration) / 1000 as AvgColdStartTimeSec, 
      count(*) as NumberOfInvocations,
      sum(@initDuration > 0) as ColdStartInvocations
by bin(1d) as TimeRange, @memorySize / 1000000 as MemoryConfigurationMB
```

<svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-on"><title>Enter fullscreen mode</title> <path d="M16 3h6v6h-2V5h-4V3zM2 3h6v2H4v4H2V3zm18 16v-4h2v6h-6v-2h4zM4 19h4v2H2v-6h2v4z"></path></svg> <svg xmlns="http://www.w3.org/2000/svg" width="20px" height="20px" viewBox="0 0 24 24" class="highlight-action crayons-icon highlight-action--fullscreen-off"><title>Exit fullscreen mode</title><path d="M18 7h4v2h-6V3h2v4zM8 9H2V7h4V3h2v6zm10 8v4h-2v-6h6v2h-4zM8 15v6H6v-4H2v-2h6z"></path></svg>

[![Rig metrics](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fob4cls9cdynuhxs1yi2s.png)](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fob4cls9cdynuhxs1yi2s.png)

> **Memory and runtime analysis**  
> At the memory configuration of **128MB**, the lambda has the lowest average memory usage of **96.1 MB** and the highest runtime of **5.1s**. At a memory configuration of **1GB**, the lambda has the highest average memory usage of **113.1 MB** and the lowest runtime of **4.4s**. In other words, with an extra ~7MB of memory usage, the lambda function was 700ms faster.
> 
> **Cold starts analysis** ❄️  
> The average initialization time remains steady around **0.16s**.

The chart below shows the power tuner results after running the app 50 times with each of the 4 memory configurations.  
[![Rig power tuner](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fczdy8596b2d8i1np9swh.png)](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fczdy8596b2d8i1np9swh.png)

We see that adding memory to the function (and therefore adding computational power) **does in fact affect the performance of the lambda by less than a second**.

### LangChain - Memory, runtime, and coldstarts
#### Deployment package

We are not able to use zip files for the deployment package of the lambda functions as the zip size exceeds the maximum size allowed by AWS. The [loader dependencies](https://github.com/garance-buricatu/rig-aws/blob/master/langchain-montreal-lancedb/loader_requirements.txt) and [app dependencies](https://github.com/garance-buricatu/rig-aws/blob/master/langchain-montreal-lancedb/app_requirements.txt) create zip files of size around 150 MB.

Instead, we must use container images. The [docker image](https://github.com/garance-buricatu/rig-aws/blob/master/langchain-montreal-lancedb/Dockerfile) has size 471.45MB using the base python lambda image.  
[![LangChain deployment package](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2F0z48s2eqw6c92hnxv79y.png)](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2F0z48s2eqw6c92hnxv79y.png)

We did the same experiment as with the Rig app above and got the following metrics:  
[![LangChain metrics](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2F2ekgng4g7iu4pxeokhap.png)](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2F2ekgng4g7iu4pxeokhap.png)

First of all, the function is **unable to run with a memory allocation of 128MB**. It gets killed at this allocation size due to lack of memory. So we will compare the three following memory configurations: 256MB, 512MB, 1GB.

> **Memory and runtime analysis**  
> At the memory configuration of **256MB**, the lambda has the lowest average memory usage of **245.8 MB** and the highest runtime of **4.9s**. At a memory configuration of **1GB**, the lambda has the highest average memory usage of **359.6 MB** and the lowest runtime of **4.0s**. In other words, with an extra **~113MB** of memory usage, the lambda function was 1s faster.
> 
> **Cold starts analysis** ❄️  
> The average initialization time increases as the memory configuration increases with the lowest being **1.9s** and the highest being **2.7s**.

The chart below shows the power tuner results after running the app 50 times with each of the 4 memory configurations.  
[![LangChain power tuner](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fjvck75kamnxfvuv3p8vi.png)](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fjvck75kamnxfvuv3p8vi.png)

We see that adding memory to the function (and therefore adding computational power) also affects the performance of the lambda by about a second.

### Final Comparison between Rig and LangChain

Based on the Cloudwatch logs produced by both the Rig and Langchain lambdas, we were able to produce the following graphics:  
[![Memory](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Ftzxvnb5fjycxe0au3fpq.png)](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Ftzxvnb5fjycxe0au3fpq.png)  
[![Cold starts](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fjptcfd6sqslrgclttd2c.png)](https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fjptcfd6sqslrgclttd2c.png)

## Resources

Rig is an emerging project in the open-source community, and we're continuously expanding its ecosystem with new integrations and tools. We believe in the power of community-driven development and welcome contributions from developers of all skill levels.

Stay connected and contribute to Rig's growth:

- 📚 [Documentation](https://docs.rs/rig-core/latest/rig/): Comprehensive guides and API references
- 💻 [GitHub Repository](https://github.com/0xPlaygrounds/rig): Contribute, report issues, or star the project
- 🌐 [Official Website](https://rig.rs/): Latest news, tutorials, and resources

Join our [community](https://discord.com/invite/playgrounds) channel to discuss ideas, seek help, and collaborate with other Rig developers.