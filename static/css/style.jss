
/* ------------------------------
   1. Global Styles
------------------------------ */

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: Arial, Helvetica, sans-serif;
    background: #f4f7f2;
    color: #26352b;
    line-height: 1.6;
}

/* ------------------------------
   2. Header
------------------------------ */

.header {
    background: #245c3a;
    color: white;
    text-align: center;
    padding: 30px 15px;
}

.header h1 {
    font-size: 30px;
    margin-bottom: 8px;
}

.header p {
    font-size: 16px;
    color: #e1eee4;
}

/* ------------------------------
   3. Main Container
------------------------------ */

.container {
    width: 90%;
    max-width: 850px;
    margin: 35px auto;
}

.intro {
    text-align: center;
    margin-bottom: 25px;
}

.intro h2 {
    color: #245c3a;
    margin-bottom: 10px;
    font-size: 25px;
}

/* ------------------------------
   4. Upload Card
------------------------------ */

.upload-card,
.result-card,
.information {
    background: white;
    padding: 28px;
    border-radius: 14px;
    margin-bottom: 25px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.07);
}

.upload-card h3,
.result-card h3,
.information h3 {
    color: #245c3a;
    margin-bottom: 18px;
}

.upload-area {
    border: 2px dashed #76a985;
    background: #f8fbf7;
    border-radius: 12px;
    min-height: 220px;
    padding: 25px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    cursor: pointer;
    transition: 0.3s;
}

.upload-area:hover {
    background: #edf6ee;
    border-color: #245c3a;
}

.upload-icon {
    font-size: 42px;
    margin-bottom: 10px;
}

.upload-area p {
    font-size: 17px;
    font-weight: bold;
    margin-bottom: 8px;
}

.upload-area span {
    font-size: 13px;
    color: #777;
}

#imagePreview {
    max-width: 100%;
    max-height: 300px;
    border-radius: 10px;
    object-fit: contain;
}

/* ------------------------------
   5. Prediction Button
------------------------------ */

#predictButton {
    display: block;
    width: 100%;
    padding: 14px;
    margin-top: 22px;
    background: #287044;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    transition: 0.3s;
}

#predictButton:hover:not(:disabled) {
    background: #194d2e;
}

#predictButton:disabled {
    background: #aab9ad;
    cursor: not-allowed;
}

.status-message {
    margin-top: 12px;
    text-align: center;
    font-size: 14px;
}

/* ------------------------------
   6. Prediction Result
------------------------------ */

.result-card {
    text-align: center;
    border-top: 5px solid #287044;
}

.result-card p {
    margin: 10px 0;
}

#gradeResult {
    color: #287044;
    font-size: 34px;
    margin: 12px 0;
}

#confidenceResult {
    color: #245c3a;
}

/* ------------------------------
   7. Information Section
------------------------------ */

.information p {
    margin-bottom: 12px;
}

.disclaimer {
    background: #fff8e8;
    color: #76551c;
    border-left: 4px solid #d9a441;
    padding: 12px;
    border-radius: 5px;
    font-size: 13px;
}

/* ------------------------------
   8. Footer
------------------------------ */

footer {
    background: #245c3a;
    color: white;
    text-align: center;
    padding: 18px;
    font-size: 13px;
}

/* ------------------------------
   9. Responsive Design
------------------------------ */

@media (max-width: 600px) {

    .header h1 {
        font-size: 22px;
    }

    .container {
        width: 94%;
        margin: 20px auto;
    }

    .upload-card,
    .result-card,
    .information {
        padding: 20px;
    }

    #gradeResult {
        font-size: 28px;
    }
}
