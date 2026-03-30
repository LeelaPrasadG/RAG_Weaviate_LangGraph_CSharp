using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Embeddings;
using UglyToad.PdfPig;
using System.Text;
using System.Numerics;
using System.IO;
using System.Net.Http;
using System.Text.Json;

#pragma warning disable SKEXP0001
#pragma warning disable SKEXP0010

public class RagService
{
    private readonly Kernel _kernel;
    private readonly ITextEmbeddingGenerationService _embeddingService;
    private readonly string _chunksFolder;
    private readonly HttpClient _httpClient;
    private readonly string _weaviateUri = "http://127.0.0.1:8080";
    private const string WEAVIATE_CLASS = "DocumentChunk";
    private List<(string Text, int ChunkIndex)> _currentChunks = new();

    public RagService()
    {
        var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        Console.WriteLine($"Initializing RagService... API Key loaded: {!string.IsNullOrEmpty(apiKey)}");
        if (string.IsNullOrEmpty(apiKey))
        {
            throw new Exception("OPENAI_API_KEY environment variable is not set.");
        }
        var builder = Kernel.CreateBuilder();        
        builder.AddOpenAIChatCompletion(modelId: "gpt-5.4", apiKey: apiKey);
        #pragma warning disable SKEXP0010 // Extension is experimental
        builder.AddOpenAITextEmbeddingGeneration(modelId: "text-embedding-3-small", apiKey: apiKey);
        _kernel = builder.Build();
        _embeddingService = _kernel.GetRequiredService<ITextEmbeddingGenerationService>();
        
        _httpClient = new HttpClient();
        _chunksFolder = Path.Combine(Directory.GetCurrentDirectory(), "WeaviateChunks");
        if (!Directory.Exists(_chunksFolder))
        {
            Directory.CreateDirectory(_chunksFolder);
            Console.WriteLine($"Created WeaviateChunks folder at {_chunksFolder}");
        }

        InitializeWeaviateSchema();
        Console.WriteLine("RagService initialized successfully.");
    }

    private void InitializeWeaviateSchema()
    {
        try
        {
            var schemaJson = @"{
              ""classes"": [{
                ""class"": """ + WEAVIATE_CLASS + @""",
                ""description"": ""Document chunks for RAG system"",
                ""properties"": [
                  {
                    ""name"": ""text"",
                    ""dataType"": [""text""],
                    ""description"": ""Chunk text content""
                  },
                  {
                    ""name"": ""chunkIndex"",
                    ""dataType"": [""int""],
                    ""description"": ""Chunk index order""
                  }
                ]
              }]
            }";

            var request = new StringContent(schemaJson, Encoding.UTF8, "application/json");
            var response = _httpClient.PostAsync($"{_weaviateUri}/v1/schema", request).Result;
            
            if (response.IsSuccessStatusCode)
            {
                Console.WriteLine($"Weaviate schema initialized for class '{WEAVIATE_CLASS}'");
            }
            else if (response.StatusCode == System.Net.HttpStatusCode.Conflict || response.StatusCode == System.Net.HttpStatusCode.BadRequest)
            {
                Console.WriteLine($"Weaviate class '{WEAVIATE_CLASS}' already exists or schema already set");
            }
            else
            {
                Console.WriteLine($"Warning: Could not initialize Weaviate schema: {response.StatusCode}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Could not initialize Weaviate schema: {ex.Message}. Make sure Weaviate is running at {_weaviateUri}");
        }
    }


    public async Task InitializeDocuments()
    {
        if (_currentChunks.Any()) return; // Already initialized
        Console.WriteLine("Initializing documents...");
        var text = "";
        var pdfDir = Path.Combine(Directory.GetCurrentDirectory(), "RAG Docs");
        
        if (!Directory.Exists(pdfDir))
        {
            Console.WriteLine($"RAG Docs folder not found at {pdfDir}");
            return;
        }

        foreach (var file in Directory.GetFiles(pdfDir, "*.pdf"))
        {
            text += ExtractTextFromPdf(file) + "\n";
        }

        if (string.IsNullOrWhiteSpace(text))
        {
            Console.WriteLine("No text extracted from PDFs");
            return;
        }

        Console.WriteLine($"Extracted text length: {text.Length}");
        var chunks = ChunkText(text, 1000);
        Console.WriteLine($"Text chunked into {chunks.Count} chunks.");

        // Clear Weaviate and local storage
        ClearWeaviateData();
        ClearChunksFolder();

        int chunkIndex = 0;
        foreach (var chunk in chunks)
        {
            try
            {
                // Sanitize chunk for storage
                var sanitizedChunk = DataSanitizer.SanitizeChunkForStorage(chunk);
                
                // Save chunk to file
                var chunkFileName = Path.Combine(_chunksFolder, $"chunk_{chunkIndex:D6}.txt");
                await File.WriteAllTextAsync(chunkFileName, sanitizedChunk);
                
                _currentChunks.Add((sanitizedChunk, chunkIndex));

                // Generate embedding
                var embedding = await _embeddingService.GenerateEmbeddingAsync(sanitizedChunk);
                
                // Store in Weaviate
                await StoreChunkInWeaviate(sanitizedChunk, chunkIndex, embedding.ToArray());
                chunkIndex++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing chunk {chunkIndex}: {ex.Message}");
            }
        }

        Console.WriteLine($"Documents initialized with {chunkIndex} chunks in Weaviate and WeaviateChunks folder.");
    }

    private async Task StoreChunkInWeaviate(string text, int chunkIndex, float[] embedding)
    {
        try
        {
            var objectJson = new
            {
                @class = WEAVIATE_CLASS,
                properties = new { text = text, chunkIndex = chunkIndex },
                vector = embedding.Cast<double>().ToList()
            };

            var content = new StringContent(JsonSerializer.Serialize(objectJson), Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync($"{_weaviateUri}/v1/objects", content);
            
            if (!response.IsSuccessStatusCode)
            {
                Console.WriteLine($"Warning: Could not store chunk {chunkIndex} in Weaviate: {response.StatusCode}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Could not store chunk {chunkIndex} in Weaviate: {ex.Message}");
        }
    }

    private void ClearWeaviateData()
    {
        try
        {
            var response = _httpClient.DeleteAsync($"{_weaviateUri}/v1/objects?class={WEAVIATE_CLASS}").Result;
            if (response.IsSuccessStatusCode)
            {
                Console.WriteLine("Cleared Weaviate data");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Could not clear Weaviate data: {ex.Message}");
        }
    }

    private void ClearChunksFolder()
    {
        try
        {
            if (Directory.Exists(_chunksFolder))
            {
                foreach (var file in Directory.GetFiles(_chunksFolder, "*.txt"))
                {
                    File.Delete(file);
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Could not clear chunks folder: {ex.Message}");
        }
    }

    public async Task<string> GetAnswer(string question)
    {
        // Sanitize user input for security
        var sanitizedQuestion = DataSanitizer.SanitizeQuestion(question);
        Console.WriteLine($"Processing question: {sanitizedQuestion}");
        
        await InitializeDocuments();
        Console.WriteLine("Documents initialized.");
        
        var questionEmbedding = await _embeddingService.GenerateEmbeddingAsync(sanitizedQuestion);
        Console.WriteLine("Question embedding generated.");
        
        var relevantChunks = await QueryWeaviateForSimilarChunks(questionEmbedding.ToArray());
        Console.WriteLine($"Retrieved {relevantChunks.Count} relevant chunks from Weaviate.");
        
        // Sanitize context to prevent prompt injection
        var sanitizedContext = DataSanitizer.SanitizeContextForPrompt(string.Join("\n", relevantChunks.Take(3)));
        var prompt = $"Context: {sanitizedContext}\nQuestion: {sanitizedQuestion}\nAnswer the question based on the context.";
        Console.WriteLine("Invoking prompt...");
        
        var response = await _kernel.InvokePromptAsync(prompt);
        var answer = response.ToString();
        Console.WriteLine($"Answer: {answer}");
        return answer;
    }

    private async Task<List<string>> QueryWeaviateForSimilarChunks(float[] embeddingVector)
    {
        var relevantChunks = new List<string>();
        
        try
        {
            var vectorList = string.Join(",", embeddingVector.Take(10).Select(v => v.ToString("F4")));
            
            var graphQLQuery = @"{
              ""query"": ""query { Get { " + WEAVIATE_CLASS + @" (nearVector: { vector: [" + vectorList + @"] }, limit: 3) { text chunkIndex } } }""
            }";

            var content = new StringContent(graphQLQuery, Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync($"{_weaviateUri}/v1/graphql", content);
            
            if (response.IsSuccessStatusCode)
            {
                var responseText = await response.Content.ReadAsStringAsync();
                // Validate response for security
                if (DataSanitizer.ValidateWeaviateResponse(responseText) && responseText.Contains("\"text\""))
                {
                    relevantChunks.Add("Retrieved content from Weaviate: " + responseText.Substring(0, Math.Min(200, responseText.Length)));
                }
                else
                {
                    Console.WriteLine("Warning: Invalid Weaviate response structure");
                    relevantChunks = _currentChunks.OrderBy(c => c.ChunkIndex).Take(3).Select(c => c.Text).ToList();
                }
            }
            else
            {
                Console.WriteLine($"Warning: Weaviate query failed: {response.StatusCode}");
                //Fallback to in-memory chunks
                relevantChunks = _currentChunks.OrderBy(c => c.ChunkIndex).Take(3).Select(c => c.Text).ToList();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Could not query Weaviate: {ex.Message}");
            // Fallback to in-memory chunks
            relevantChunks = _currentChunks.OrderBy(c => c.ChunkIndex).Take(3).Select(c => c.Text).ToList();
        }

        return relevantChunks;
    }

    private string ExtractTextFromPdf(string path)
    {
        try
        {
            using var document = PdfDocument.Open(path);
            var sb = new StringBuilder();
            foreach (var page in document.GetPages())
            {
                sb.Append(page.Text);
            }
            // Sanitize extracted PDF content
            var sanitized = DataSanitizer.SanitizePdfContent(sb.ToString());
            return sanitized;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: could not extract text from PDF '{path}': {ex.Message}");
            return string.Empty;
        }
    }

    private List<string> ChunkText(string text, int chunkSize)
    {
        var chunks = new List<string>();
        for (int i = 0; i < text.Length; i += chunkSize)
        {
            chunks.Add(text.Substring(i, Math.Min(chunkSize, text.Length - i)));
        }
        return chunks;
    }
}