function analyzeText() {
    let input = document.getElementById("userInput").value;
    let resultDiv = document.getElementById("result");
  
    if (!input) {
      resultDiv.innerHTML = "<p>Please enter a message.</p>";
      return;
    }
  
    // Mock categories & random confidence score
    let categories = ["Hate Speech", "Spam", "Safe", "Harassment"];
    let randomCategory = categories[Math.floor(Math.random() * categories.length)];
    let confidence = (Math.random() * (0.9 - 0.7) + 0.7).toFixed(2);
  
    resultDiv.innerHTML = `
      <h3>Analysis Result:</h3>
      <p><strong>Category:</strong> ${randomCategory}</p>
      <p><strong>Confidence:</strong> ${confidence}</p>
      <p><i>In real deployment, this would trigger bot action on Telegram/X/Instagram.</i></p>
    `;
  }