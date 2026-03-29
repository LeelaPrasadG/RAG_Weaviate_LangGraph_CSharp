using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Embeddings;
using UglyToad.PdfPig;
using System.Text;
using System.Numerics;
using System.IO;

#pragma warning disable SKEXP0001
#pragma warning disable SKEXP0010

public class RagService
{
    private readonly Kernel _kernel;
    private readonly ITextEmbeddingGenerationService _embeddingService;
    private List<(string Text, ReadOnlyMemory<float> Embedding)> _documents = new();

    public RagService()
    {
        var apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY");
        Console.WriteLine($"Initializing RagService... and key is Hidden for security reasons.");
        Console.WriteLine($"API Key loaded: {!string.IsNullOrEmpty(apiKey)}");
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
        Console.WriteLine("RagService initialized successfully.");
    }

    public async Task InitializeDocuments()
    {
        if (_documents.Any()) return; // Already initialized
        Console.WriteLine("Initializing documents...");
        var text = "";
        var pdfDir = Path.Combine(Directory.GetCurrentDirectory(), "RAG Docs");
        foreach (var file in Directory.GetFiles(pdfDir, "*.pdf"))
        {
            text += ExtractTextFromPdf(file) + "\n";
        }
        //text = "This is a sample document for testing the RAG service. It contains information about various topics that can be used to answer questions. The RAG service will extract relevant chunks of this document to provide accurate answers based on the context.";
        Console.WriteLine($"Extracted text length: {text.Length} and text is {text.Substring(0, Math.Min(500, text.Length))}...");
        var chunks = ChunkText(text, 1000);
        Console.WriteLine($"Text chunked into {chunks.Count} chunks.");
        foreach (var chunk in chunks)
        {
            var embedding = await _embeddingService.GenerateEmbeddingAsync(chunk);
            _documents.Add((chunk, embedding));
        }
        Console.WriteLine($"Documents initialized with {_documents.Count} chunks.");
    }

    public async Task<string> GetAnswer(string question)
    {
        Console.WriteLine($"Processing question: {question}");
        await InitializeDocuments();
        Console.WriteLine("Documents initialized.");
        var questionEmbedding = await _embeddingService.GenerateEmbeddingAsync(question);
        Console.WriteLine("Question embedding generated.");
        var similarities = _documents.Select(d => (d.Text, CosineSimilarity(d.Embedding.Span, questionEmbedding.Span))).ToList();
        var top3 = similarities.OrderByDescending(s => s.Item2).Take(3).Select(s => s.Text).ToList();
        Console.WriteLine($"Top 3 chunks selected: {top3.Count}");
        var context = string.Join("\n", top3);
        var prompt = $"Context: {context}\nQuestion: {question}\nAnswer the question based on the context.";
        Console.WriteLine("Invoking prompt...");
        var response = await _kernel.InvokePromptAsync(prompt);
        var answer = response.ToString();
        Console.WriteLine($"Answer: {answer}");
        return answer;
    }

    private static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (MathF.Sqrt(normA) * MathF.Sqrt(normB));
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
            return sb.ToString();
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