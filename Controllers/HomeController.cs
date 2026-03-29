using Microsoft.AspNetCore.Mvc;

namespace RAG_Weaviate_LangGraph_CSharp.Controllers
{
    public class HomeController : Controller
    {
        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> Ask(string question)
        {
            // Hardcode question for testing
            //question = "What is the Transformer model?";
            var ragService = new RagService();
            string answer = await ragService.GetAnswer(question);
            ViewBag.Answer = answer;
            return View("Index");
        }
    }
}