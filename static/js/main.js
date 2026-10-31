(function () {
    const dashboardNode = document.getElementById("dashboard-data");
    if (dashboardNode && window.Chart) {
        const payload = JSON.parse(dashboardNode.textContent);
        const classCounts = payload.class_counts || {};
        const classCanvas = document.getElementById("classChart");
        if (classCanvas) {
            new Chart(classCanvas, {
                type: "bar",
                data: {
                    labels: Object.keys(classCounts),
                    datasets: [{
                        label: "Images",
                        data: Object.values(classCounts),
                        backgroundColor: "#1f5c45"
                    }]
                },
                options: { plugins: { legend: { display: false } } }
            });
        }
        const modelCanvas = document.getElementById("modelChart");
        if (modelCanvas) {
            const rows = payload.comparison || [];
            new Chart(modelCanvas, {
                type: "bar",
                data: {
                    labels: rows.map((row) => row.model),
                    datasets: [{
                        label: "Test accuracy (%)",
                        data: rows.map((row) => row.accuracy == null ? 0 : row.accuracy * 100),
                        backgroundColor: "#c9842a"
                    }]
                },
                options: { indexAxis: "y", plugins: { legend: { display: false } } }
            });
        }
    }

    const uploadArea = document.getElementById("uploadArea");
    const imageInput = document.getElementById("imageInput");
    const imagePreview = document.getElementById("imagePreview");
    const uploadContent = document.getElementById("uploadContent");
    const predictButton = document.getElementById("predictButton");
    const statusMessage = document.getElementById("statusMessage");
    const resultCard = document.getElementById("resultCard");
    if (!uploadArea || !imageInput) {
        return;
    }

    let selectedImage = null;

    uploadArea.addEventListener("click", () => imageInput.click());
    imageInput.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (!file) {
            return;
        }
        if (!["image/jpeg", "image/png"].includes(file.type)) {
            statusMessage.textContent = "Please select a JPG or PNG image.";
            predictButton.disabled = true;
            return;
        }
        selectedImage = file;
        imagePreview.src = URL.createObjectURL(file);
        imagePreview.classList.remove("d-none");
        if (uploadContent) {
            uploadContent.classList.add("d-none");
        }
        predictButton.disabled = false;
        statusMessage.textContent = "Image ready.";
    });

    predictButton.addEventListener("click", async () => {
        if (!selectedImage) {
            return;
        }
        const formData = new FormData();
        formData.append("image", selectedImage);
        predictButton.disabled = true;
        statusMessage.textContent = "Estimating grade from image appearance...";
        try {
            const response = await fetch("/api/predict", { method: "POST", body: formData });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || "Prediction failed.");
            }
            document.getElementById("gradeResult").textContent = data.grade;
            document.getElementById("confidenceResult").textContent =
                data.confidence_percent + "% (uncalibrated)";
            const value = data.estimated_acrylamide_ug_per_kg;
            document.getElementById("valueResult").textContent =
                value == null ? "not available" : value + " µg/kg (image estimate)";
            const bars = document.getElementById("confidenceBars");
            bars.innerHTML = "";
            Object.entries(data.class_confidences || {}).forEach(([name, score]) => {
                const wrap = document.createElement("div");
                wrap.innerHTML =
                    `<div class="d-flex justify-content-between small"><span>${name}</span><span>${score}%</span></div>` +
                    `<div class="confidence-bar"><span style="width:${score}%"></span></div>`;
                bars.appendChild(wrap);
            });
            resultCard.classList.remove("d-none");
            statusMessage.textContent = "Estimate complete. Not a laboratory result.";
        } catch (error) {
            statusMessage.textContent = error.message;
        } finally {
            predictButton.disabled = false;
        }
    });
})();
