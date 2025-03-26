from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

@app.route('/')
def home():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple Symptom Checker</title>
        <style>
            body { font-family: Arial; margin: 20px; }
            .container { max-width: 600px; margin: 0 auto; }
            .result { margin-top: 20px; padding: 10px; background: #f0f0f0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Simple Symptom Checker</h1>
            <div>
                <input type="text" id="symptom" placeholder="Enter a symptom">
                <button onclick="addSymptom()">Add</button>
            </div>
            <div id="symptoms-list" style="margin-top: 10px;"></div>
            <button onclick="analyzeSymptoms()" style="margin-top: 10px;">Analyze</button>
            <div id="result" class="result" style="display: none;"></div>
        </div>
        
        <script>
            const symptoms = [];
            
            function addSymptom() {
                const symptomInput = document.getElementById('symptom');
                const symptom = symptomInput.value.trim();
                
                if (symptom && !symptoms.includes(symptom)) {
                    symptoms.push(symptom);
                    updateSymptomsList();
                    symptomInput.value = '';
                }
            }
            
            function updateSymptomsList() {
                const list = document.getElementById('symptoms-list');
                list.innerHTML = '';
                
                symptoms.forEach(symptom => {
                    const div = document.createElement('div');
                    div.textContent = symptom;
                    list.appendChild(div);
                });
            }
            
            function analyzeSymptoms() {
                if (symptoms.length === 0) {
                    alert('Please add at least one symptom');
                    return;
                }
                
                fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ symptoms })
                })
                .then(response => response.json())
                .then(data => {
                    const resultDiv = document.getElementById('result');
                    resultDiv.style.display = 'block';
                    resultDiv.innerHTML = `<h3>Possible Condition: ${data.condition}</h3>
                                          <p>Recommendation: ${data.recommendation}</p>`;
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred');
                });
            }
        </script>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    symptoms = data.get('symptoms', [])
    
    # Very simple "analysis"
    if 'headache' in symptoms:
        condition = 'Possible Migraine'
        recommendation = 'Rest in a dark room and consider over-the-counter pain relievers'
    elif 'fever' in symptoms:
        condition = 'Possible Cold or Flu'
        recommendation = 'Rest, stay hydrated, and monitor temperature'
    elif 'cough' in symptoms:
        condition = 'Possible Respiratory Infection'
        recommendation = 'Rest, stay hydrated, and consider cough medicine'
    else:
        condition = 'Unknown Condition'
        recommendation = 'Monitor symptoms and consult a doctor if they worsen'
    
    return jsonify({
        'condition': condition,
        'recommendation': recommendation
    })

if __name__ == '__main__':
    app.run(debug=True) 