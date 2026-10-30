
// ------------------------------------
// 1. Get HTML elements
// ------------------------------------

const imageInput = document.getElementById("imageInput");
const uploadArea = document.getElementById("uploadArea");
const uploadContent = document.getElementById("uploadContent");
const imagePreview = document.getElementById("imagePreview");

const predictButton = document.getElementById("predictButton");
const statusMessage = document.getElementById("statusMessage");

const resultCard = document.getElementById("resultCard");
const gradeResult = document.getElementById("gradeResult");
const confidenceResult = document.getElementById("confidenceResult");
const valueResult = document.getElementById("valueResult");
const predictionDisclaimer = document.getElementById("predictionDisclaimer");

let selectedImage = null;


// ------------------------------------
// 2. Open file selection
// ------------------------------------

uploadArea.addEventListener("click", () => {
    imageInput.click();
});


// ------------------------------------
// 3. Preview selected image
// ------------------------------------

imageInput.addEventListener("change", (event) => {

    const file = event.target.files[0];

    if (!file) {
        return;
    }

    const allowedTypes = [
        "image/jpeg",
        "image/png"
    ];

    if (!allowedTypes.includes(file.type)) {

        statusMessage.textContent =
            "Please select a JPG, JPEG or PNG image.";

        statusMessage.style.color = "red";

        imageInput.value = "";
        selectedImage = null;
        predictButton.disabled = true;

        return;
    }

    selectedImage = file;

    const imageURL = URL.createObjectURL(file);

    imagePreview.src = imageURL;
    imagePreview.style.display = "block";

    uploadContent.style.display = "none";

    predictButton.disabled = false;

    statusMessage.textContent =
        "Image selected successfully. Click Predict.";

    statusMessage.style.color = "green";

    resultCard.style.display = "none";
});


// ------------------------------------
// 4. Send image to Flask backend
// ------------------------------------

predictButton.addEventListener("click", async () => {

    if (!selectedImage) {

        statusMessage.textContent =
            "Please select an image first.";

        return;
    }

    const formData = new FormData();

    formData.append("image", selectedImage);

    predictButton.disabled = true;

    predictButton.textContent = "Analyzing Image...";

    statusMessage.textContent =
        "Please wait while both AI models analyze your image.";

    statusMessage.style.color = "#287044";

    resultCard.style.display = "none";

    try {

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "Prediction failed."
            );
        }

        // ------------------------------------
        // 5. Display classification result
        // ------------------------------------

        gradeResult.textContent = data.grade;

        confidenceResult.textContent =
            data.confidence + "%";


        // ------------------------------------
        // 6. Display numerical regression result
        // ------------------------------------

        valueResult.textContent =
            Number(data.acrylamide_value).toFixed(2);

        if (predictionDisclaimer && data.disclaimer) {
            predictionDisclaimer.textContent =
                data.disclaimer;
        }

        // Show results
        resultCard.style.display = "block";

        statusMessage.textContent =
            "Prediction completed successfully!";

        statusMessage.style.color = "green";

    } catch (error) {

        statusMessage.textContent = error.message;

        statusMessage.style.color = "red";

    } finally {

        predictButton.disabled = false;

        predictButton.textContent =
            "Predict Acrylamide Grade and Value";
    }
});
