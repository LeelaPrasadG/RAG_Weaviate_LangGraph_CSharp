using System;
using System.Text;
using System.Text.RegularExpressions;

/// <summary>
/// Sanitizes user input, PDF content, and LLM prompts to prevent injection attacks
/// and ensure data integrity in the RAG pipeline.
/// </summary>
public class DataSanitizer
{
    /// <summary>Sanitizes user questions to prevent prompt injection.</summary>
    public static string SanitizeQuestion(string input)
    {
        if (string.IsNullOrWhiteSpace(input))
            return string.Empty;

        // Limit length to prevent DoS
        var sanitized = input.Length > 2000 ? input.Substring(0, 2000) : input;

        // Remove control characters
        sanitized = RemoveControlCharacters(sanitized);

        // Escape special characters for safety
        sanitized = EscapeSpecialCharacters(sanitized);

        return sanitized.Trim();
    }

    /// <summary>Sanitizes extracted PDF text by removing harmful content.</summary>
    public static string SanitizePdfContent(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return string.Empty;

        // Remove null bytes and control characters (common in PDFs)
        var sanitized = text.Replace("\0", "");
        sanitized = RemoveControlCharacters(sanitized);

        // Remove excessive whitespace
        sanitized = Regex.Replace(sanitized, @"\s+", " ");

        // Remove URLs to prevent injection via links
        sanitized = Regex.Replace(sanitized, @"https?://[^\s]+", "[URL_REMOVED]");

        // Remove email addresses
        sanitized = Regex.Replace(sanitized, @"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL_REMOVED]");

        return sanitized;
    }

    /// <summary>Sanitizes chunk text before storage in Weaviate and files.</summary>
    public static string SanitizeChunkForStorage(string chunk)
    {
        if (string.IsNullOrWhiteSpace(chunk))
            return string.Empty;

        var sanitized = SanitizePdfContent(chunk);

        // Ensure JSON-safe for Weaviate storage
        sanitized = EscapeJsonString(sanitized);

        return sanitized;
    }

    /// <summary>Sanitizes context before passing to LLM prompt to prevent prompt injection.</summary>
    public static string SanitizeContextForPrompt(string context)
    {
        if (string.IsNullOrWhiteSpace(context))
            return string.Empty;

        // Remove any newlines that could break the prompt structure
        var sanitized = context.Replace("\n", " ").Replace("\r", " ");

        // Escape quotes that could break the prompt
        sanitized = sanitized.Replace("\"", "\\\"").Replace("'", "\\'");

        // Remove prompt injection keywords
        var injectionKeywords = new[] { "ignore instructions", "system prompt", "jailbreak", "override", "disregard" };
        foreach (var keyword in injectionKeywords)
        {
            sanitized = Regex.Replace(sanitized, Regex.Escape(keyword), "[REDACTED]", RegexOptions.IgnoreCase);
        }

        return sanitized;
    }

    /// <summary>Validates and sanitizes Weaviate API responses.</summary>
    public static bool ValidateWeaviateResponse(string response)
    {
        if (string.IsNullOrWhiteSpace(response))
            return false;

        // Check for error indicators
        if (response.Contains("\"errors\"") || response.Contains("\"error\""))
            return false;

        // Ensure it's valid JSON-like structure
        if (!response.Contains("\"data\""))
            return false;

        return true;
    }

    /// <summary>Removes control and non-printable characters.</summary>
    private static string RemoveControlCharacters(string text)
    {
        if (string.IsNullOrEmpty(text))
            return text;

        var sb = new StringBuilder();
        foreach (char c in text)
        {
            // Keep printable characters and common whitespace
            if (!char.IsControl(c) || c == '\n' || c == '\r' || c == '\t')
            {
                sb.Append(c);
            }
        }
        return sb.ToString();
    }

    /// <summary>Escapes special characters for string safety.</summary>
    private static string EscapeSpecialCharacters(string text)
    {
        if (string.IsNullOrEmpty(text))
            return text;

        return text
            .Replace("\\", "\\\\")
            .Replace("'", "''")
            .Replace("\"", "\\\"");
    }

    /// <summary>Escapes characters for JSON string safety.</summary>
    private static string EscapeJsonString(string text)
    {
        if (string.IsNullOrEmpty(text))
            return text;

        return text
            .Replace("\\", "\\\\")
            .Replace("\"", "\\\"")
            .Replace("\n", "\\n")
            .Replace("\r", "\\r")
            .Replace("\t", "\\t");
    }
}
