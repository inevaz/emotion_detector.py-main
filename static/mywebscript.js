let RunSentimentAnalysis = () => {
    let textToAnalyze = document.getElementById("textToAnalyze").value;

    let xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            document.getElementById("system_response").innerHTML = xhttp.responseText;
        }
    };
    xhttp.open("GET", "http://127.0.0.1:5000/emotionDetector?textToAnalyze=" + encodeURIComponent(textToAnalyze), true);
    xhttp.send();
}
