using HybridSearch.Models;

namespace HybridSearch.Tests.TestData;

/// <summary>
/// Generates synthetic document corpora for testing.
/// All generation is deterministic (seeded RNG) for reproducible tests.
/// Documents have realistic technical content across multiple topics with
/// deterministic fake embeddings based on topic affinity.
/// </summary>
public static class SyntheticCorpusGenerator
{
    /// <summary>
    /// Topic categories for generating diverse, realistic technical content.
    /// </summary>
    private static readonly (string Name, string[] Phrases)[] Topics =
    [
        ("Machine Learning", [
            "neural networks learn complex patterns from training data",
            "gradient descent optimizes model parameters iteratively",
            "convolutional layers extract spatial features from images",
            "transformer architecture enables parallel sequence processing",
            "overfitting occurs when models memorize training examples",
            "regularization techniques prevent model overfitting",
            "batch normalization stabilizes deep network training",
            "attention mechanisms weigh input sequence elements dynamically",
            "transfer learning adapts pretrained models to new tasks",
            "hyperparameter tuning affects model accuracy significantly",
            "recurrent neural networks process sequential data effectively",
            "generative adversarial networks create realistic synthetic data",
            "reinforcement learning agents maximize cumulative reward",
            "feature engineering transforms raw data into informative signals",
            "cross-validation estimates model generalization performance"
        ]),
        ("Cloud Infrastructure", [
            "kubernetes orchestrates containerized application deployments",
            "microservices decompose monoliths into independent services",
            "load balancers distribute traffic across server instances",
            "autoscaling adjusts compute capacity based on demand",
            "infrastructure as code provisions resources declaratively",
            "service mesh manages inter-service communication securely",
            "container images package applications with dependencies",
            "horizontal pod autoscaler adjusts replica counts dynamically",
            "ingress controllers route external traffic to services",
            "persistent volumes provide durable storage for stateful workloads",
            "helm charts template kubernetes resource manifests",
            "virtual machines provide isolated compute environments",
            "serverless functions execute code without managing servers",
            "cloud-native applications leverage platform scalability features",
            "multi-region deployments ensure high availability globally"
        ]),
        ("Database Systems", [
            "SQL queries retrieve structured data from relational tables",
            "indexing accelerates query performance on large datasets",
            "ACID transactions ensure data consistency and durability",
            "sharding distributes data across multiple database nodes",
            "replication provides fault tolerance and read scalability",
            "query optimization transforms logical plans into efficient execution",
            "denormalization trades storage for improved read performance",
            "connection pooling reduces database connection overhead",
            "stored procedures encapsulate business logic server-side",
            "materialized views precompute expensive aggregate queries",
            "write-ahead logging ensures crash recovery for databases",
            "column-oriented storage improves analytical query throughput",
            "graph databases model highly connected entity relationships",
            "document stores provide flexible schema-free data storage",
            "time-series databases optimize temporal data ingestion"
        ]),
        ("Security", [
            "encryption protects sensitive data at rest and in transit",
            "authentication verifies user identity with credentials",
            "authorization controls resource access based on permissions",
            "TLS certificates establish encrypted communication channels",
            "vulnerability scanning identifies security weaknesses proactively",
            "penetration testing simulates attacks to evaluate defenses",
            "security tokens provide stateless authentication mechanisms",
            "role-based access control assigns permissions by user role",
            "secrets management securely stores API keys and passwords",
            "zero-trust architecture verifies every access request explicitly",
            "intrusion detection systems monitor network traffic anomalies",
            "multi-factor authentication adds additional verification layers",
            "security information event management aggregates log data",
            "data loss prevention monitors sensitive information flows",
            "web application firewalls filter malicious HTTP traffic"
        ]),
        ("DevOps", [
            "continuous integration automates build and test pipelines",
            "continuous deployment releases validated changes automatically",
            "monitoring dashboards visualize system health metrics",
            "alerting rules notify teams of operational anomalies",
            "log aggregation centralizes distributed application logs",
            "infrastructure provisioning automates environment creation",
            "configuration management maintains consistent server states",
            "canary deployments validate changes with limited traffic",
            "blue-green deployments enable zero-downtime releases",
            "chaos engineering tests system resilience under failure",
            "GitOps manages infrastructure through version-controlled manifests",
            "observability combines metrics logs and traces for insights",
            "incident response procedures minimize service disruption impact",
            "capacity planning predicts future resource requirements",
            "site reliability engineering balances feature velocity with stability"
        ]),
        ("Programming Languages", [
            "static typing catches errors at compile time before runtime",
            "garbage collection automatically reclaims unused memory allocations",
            "pattern matching enables concise conditional data extraction",
            "async await simplifies asynchronous programming patterns",
            "generics enable type-safe reusable data structure implementations",
            "closures capture surrounding scope variables for deferred execution",
            "immutability prevents accidental state modification in programs",
            "type inference reduces boilerplate without sacrificing safety",
            "algebraic data types model domain concepts precisely",
            "first-class functions enable higher-order programming patterns",
            "null safety eliminates null reference exceptions statically",
            "coroutines provide lightweight cooperative multitasking",
            "trait-based polymorphism enables flexible code composition",
            "expression-bodied members provide concise syntax shortcuts",
            "destructuring assignment extracts values from complex structures"
        ])
    ];

    /// <summary>
    /// Generate a small corpus (5 documents) for basic unit tests.
    /// Each document covers a distinct topic.
    /// </summary>
    public static List<Document> GenerateSmall(int embeddingDims = 128)
    {
        return GenerateCorpus(5, embeddingDims, seed: 42);
    }

    /// <summary>
    /// Generate a medium corpus (100 documents) for more thorough tests.
    /// </summary>
    public static List<Document> GenerateMedium(int embeddingDims = 128)
    {
        return GenerateCorpus(100, embeddingDims, seed: 42);
    }

    /// <summary>
    /// Generate a large corpus (3000 documents) for performance and scale tests.
    /// </summary>
    public static List<Document> GenerateLarge(int embeddingDims = 768)
    {
        return GenerateCorpus(3000, embeddingDims, seed: 42);
    }

    /// <summary>
    /// Generate a corpus of the specified size with deterministic content and embeddings.
    /// </summary>
    /// <param name="count">Number of documents to generate.</param>
    /// <param name="embeddingDims">Embedding vector dimensions.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public static List<Document> GenerateCorpus(int count, int embeddingDims, int seed)
    {
        var rng = new Random(seed);
        var docs = new List<Document>(count);

        for (int i = 0; i < count; i++)
        {
            int primaryTopic = i % Topics.Length;
            int secondaryTopic = (i + 1 + rng.Next(Topics.Length - 1)) % Topics.Length;

            string body = GenerateBody(rng, primaryTopic, secondaryTopic);
            string title = GenerateTitle(rng, primaryTopic, i);
            float[] embedding = GenerateEmbedding(rng, embeddingDims, primaryTopic);

            docs.Add(new Document
            {
                Id = $"doc-{i:D5}",
                Title = title,
                Body = body,
                Embedding = embedding,
                Metadata = new Dictionary<string, string>
                {
                    ["topic"] = Topics[primaryTopic].Name,
                    ["index"] = i.ToString(),
                    ["source"] = "synthetic"
                }
            });
        }

        return docs;
    }

    /// <summary>
    /// Generate a document body by combining phrases from a primary and secondary topic.
    /// This creates realistic multi-sentence documents with cross-topic overlap.
    /// </summary>
    private static string GenerateBody(Random rng, int primaryTopic, int secondaryTopic)
    {
        var phrases = Topics[primaryTopic].Phrases;
        var secondaryPhrases = Topics[secondaryTopic].Phrases;

        // Pick 3-5 primary phrases and 1-2 secondary phrases
        int primaryCount = 3 + rng.Next(3);
        int secondaryCount = 1 + rng.Next(2);

        var sentences = new List<string>();

        for (int j = 0; j < primaryCount; j++)
        {
            int idx = rng.Next(phrases.Length);
            sentences.Add(CapitalizeFirst(phrases[idx]) + ".");
        }

        for (int j = 0; j < secondaryCount; j++)
        {
            int idx = rng.Next(secondaryPhrases.Length);
            sentences.Add(CapitalizeFirst(secondaryPhrases[idx]) + ".");
        }

        // Shuffle sentences for natural feel
        for (int j = sentences.Count - 1; j > 0; j--)
        {
            int k = rng.Next(j + 1);
            (sentences[j], sentences[k]) = (sentences[k], sentences[j]);
        }

        return string.Join(" ", sentences);
    }

    /// <summary>
    /// Generate a document title based on topic and index.
    /// </summary>
    private static string GenerateTitle(Random rng, int topicIndex, int docIndex)
    {
        string[] prefixes = ["Guide to", "Understanding", "Introduction to", "Deep Dive into",
            "Best Practices for", "Overview of", "Fundamentals of", "Advanced"];

        string prefix = prefixes[rng.Next(prefixes.Length)];
        string topic = Topics[topicIndex].Name;
        return $"{prefix} {topic} #{docIndex}";
    }

    /// <summary>
    /// Generate a deterministic embedding vector with topic-based signal.
    /// Documents on the same primary topic will have higher cosine similarity.
    /// The embedding is a blend of a topic "centroid" direction and random noise.
    /// </summary>
    private static float[] GenerateEmbedding(Random rng, int dims, int topicIndex)
    {
        var embedding = new float[dims];

        // Create a topic-specific signal: each topic gets a different region of the vector
        // that has stronger values, creating natural clustering
        int topicOffset = (topicIndex * dims) / Topics.Length;
        int topicSpan = dims / Topics.Length;

        for (int d = 0; d < dims; d++)
        {
            // Base: small random noise
            float noise = (float)(rng.NextDouble() * 0.3 - 0.15);

            // Topic signal: stronger values in the topic's region
            float signal = 0f;
            if (d >= topicOffset && d < topicOffset + topicSpan)
            {
                signal = (float)(0.5 + rng.NextDouble() * 0.5);
            }

            embedding[d] = noise + signal;
        }

        // Normalize to unit length
        float norm = 0f;
        for (int d = 0; d < dims; d++)
            norm += embedding[d] * embedding[d];
        norm = MathF.Sqrt(norm);

        if (norm > 0)
        {
            for (int d = 0; d < dims; d++)
                embedding[d] /= norm;
        }

        return embedding;
    }

    private static string CapitalizeFirst(string s)
    {
        if (string.IsNullOrEmpty(s)) return s;
        return char.ToUpperInvariant(s[0]) + s[1..];
    }
}
