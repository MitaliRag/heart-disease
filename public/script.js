
const form = document.getElementById("predictionForm");
const button = document.getElementById("predictButton");
const statusText = document.getElementById("status");
const resultBox = document.getElementById("result");
const resultText = document.getElementById("resultText");
const probabilityText = document.getElementById("probability");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const formData = new FormData(form);
  const data = {};

  for (const [key, value] of formData.entries()) {
    data[key] = Number(value);
  }

  button.disabled = true;
  button.textContent = "Predicting...";
  statusText.textContent = "Please wait while the model processes your data.";
  resultBox.hidden = true;

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(data)
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Prediction request failed.");
    }

    resultText.textContent = result.result;

    probabilityText.textContent =
      result.probability == null
        ? ""
        : `Model-estimated probability for class 1: ${
            (result.probability * 100).toFixed(2)
          }%`;

    resultBox.hidden = false;
    statusText.textContent = "Prediction completed.";
  } catch (error) {
    statusText.textContent = `Error: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "Predict";
  }
});
