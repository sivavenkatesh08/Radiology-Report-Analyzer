"""
Medical Knowledge Base
-----------------------
Curated medical knowledge documents for RAG retrieval.
Each entry contains a topic, content, severity guidance, and recommendations.
These are indexed into FAISS for semantic similarity search.
"""

MEDICAL_KNOWLEDGE = [
    # ── PULMONARY CONDITIONS ──
    {
        "topic": "Pulmonary Opacity and Infiltrate",
        "content": (
            "Pulmonary opacity refers to any area in the lung that appears denser than normal on imaging. "
            "Opacities can be ground-glass (hazy, does not obscure vessels), consolidation (dense, obscures vessels), "
            "or mixed. Common causes include pneumonia, pulmonary edema, hemorrhage, and atelectasis. "
            "Ground-glass opacities (GGO) are often seen in viral pneumonias, early bacterial infections, "
            "and interstitial lung diseases. Consolidation indicates alveolar filling with fluid, pus, or cells."
        ),
        "severity": "Medium",
        "recommendation": (
            "Clinical correlation with symptoms is essential. For new opacities, consider CBC, CRP, "
            "and sputum culture. Follow-up imaging in 2-4 weeks to assess resolution. "
            "If persistent, consider CT for further characterization."
        ),
    },
    {
        "topic": "Pneumonia — Community Acquired",
        "content": (
            "Community-acquired pneumonia (CAP) presents as focal consolidation or patchy opacities on chest X-ray. "
            "Lobar pneumonia shows dense consolidation confined to one lobe with air bronchograms. "
            "Bronchopneumonia shows patchy, multifocal opacities. Interstitial pneumonia shows reticular or "
            "ground-glass pattern. Complications include parapneumonic effusion, empyema, and lung abscess. "
            "Typical organisms: Streptococcus pneumoniae, Haemophilus influenzae, Mycoplasma pneumoniae."
        ),
        "severity": "Medium to High",
        "recommendation": (
            "Start empiric antibiotics per local guidelines. Obtain blood cultures if hospitalized. "
            "Follow-up chest X-ray in 4-6 weeks to confirm resolution. Consider CT if no improvement "
            "or suspicion of underlying malignancy."
        ),
    },
    {
        "topic": "Pleural Effusion",
        "content": (
            "Pleural effusion is fluid accumulation in the pleural space. On upright chest X-ray, it appears as "
            "blunting of the costophrenic angle (requires >200mL). Lateral decubitus view can detect smaller amounts. "
            "Causes: transudative (heart failure, cirrhosis, nephrotic syndrome) or exudative (infection, malignancy, "
            "PE, autoimmune). Massive effusion can cause mediastinal shift and respiratory compromise. "
            "Light's criteria distinguish transudative from exudative effusions."
        ),
        "severity": "Medium",
        "recommendation": (
            "Determine if transudative or exudative using Light's criteria via thoracentesis. "
            "Treat underlying cause. Large or symptomatic effusions may require therapeutic drainage. "
            "Send fluid for cell count, protein, LDH, glucose, pH, cytology, and culture."
        ),
    },
    {
        "topic": "Pneumothorax",
        "content": (
            "Pneumothorax is air in the pleural space causing partial or complete lung collapse. "
            "On chest X-ray: visible visceral pleural line with absence of lung markings peripherally. "
            "Types: spontaneous (primary in tall, thin young adults; secondary in COPD/asthma), "
            "traumatic, and tension pneumothorax (medical emergency with mediastinal shift). "
            "Tension pneumothorax shows contralateral mediastinal shift and hemodynamic instability."
        ),
        "severity": "High",
        "recommendation": (
            "Small pneumothorax (<2cm): observation with serial X-rays. "
            "Large or symptomatic: needle aspiration or chest tube insertion. "
            "Tension pneumothorax: immediate needle decompression at 2nd intercostal space, "
            "mid-clavicular line, followed by chest tube."
        ),
    },
    {
        "topic": "Pulmonary Nodule",
        "content": (
            "Pulmonary nodule is a round opacity ≤3cm surrounded by pulmonary parenchyma. "
            "Solitary pulmonary nodules (SPN) are found in 0.09-0.2% of chest X-rays. "
            "Benign features: calcification (central, popcorn, laminar), stable size >2 years, smooth margins. "
            "Malignant features: spiculated margins, upper lobe location, >8mm size, growth on serial imaging. "
            "Fleischner Society guidelines provide follow-up recommendations based on size and risk factors."
        ),
        "severity": "Medium to High",
        "recommendation": (
            "Apply Fleischner Society guidelines. For nodules >8mm in high-risk patients, "
            "consider PET/CT or biopsy. Compare with prior imaging. Low-risk small nodules (<6mm) "
            "may not require follow-up. Document size, morphology, and location."
        ),
    },
    # ── CARDIAC CONDITIONS ──
    {
        "topic": "Cardiomegaly",
        "content": (
            "Cardiomegaly is enlargement of the cardiac silhouette, defined as cardiothoracic ratio >0.5 "
            "on PA chest X-ray. Causes include dilated cardiomyopathy, valvular heart disease, "
            "pericardial effusion, hypertensive heart disease, and ischemic cardiomyopathy. "
            "Associated findings may include pulmonary venous congestion, cephalization of vessels, "
            "Kerley B lines, and pleural effusions suggesting heart failure."
        ),
        "severity": "Medium",
        "recommendation": (
            "Echocardiogram to assess chamber sizes, ejection fraction, and valvular function. "
            "BNP/NT-proBNP levels. Cardiology referral. Evaluate for treatable causes: "
            "hypertension, valvular disease, coronary artery disease, thyroid dysfunction."
        ),
    },
    {
        "topic": "Pulmonary Edema",
        "content": (
            "Pulmonary edema is excess fluid in the lung parenchyma and alveoli. "
            "Cardiogenic: cephalization of pulmonary vessels, peribronchial cuffing, Kerley B lines, "
            "bilateral alveolar opacities (bat-wing pattern), pleural effusions. "
            "Non-cardiogenic (ARDS): diffuse bilateral opacities without cardiomegaly or effusions. "
            "Stages: redistribution → interstitial edema → alveolar flooding."
        ),
        "severity": "High",
        "recommendation": (
            "Cardiogenic: IV diuretics, oxygen, nitrates, positive pressure ventilation if needed. "
            "Identify and treat precipitating cause. Monitor with daily chest X-rays. "
            "Echocardiogram to assess cardiac function. Consider ICU admission for severe cases."
        ),
    },
    # ── SKELETAL CONDITIONS ──
    {
        "topic": "Rib Fracture",
        "content": (
            "Rib fractures are breaks in the rib bones, most commonly ribs 4-9. "
            "On chest X-ray: visible cortical disruption, but up to 50% may be occult on initial films. "
            "Complications: pneumothorax, hemothorax, flail chest (≥3 consecutive ribs fractured in 2 places), "
            "pulmonary contusion. Lower rib fractures may be associated with hepatic or splenic injury. "
            "Pathologic fractures suggest metastatic disease."
        ),
        "severity": "Medium",
        "recommendation": (
            "Pain management is primary treatment. Incentive spirometry to prevent atelectasis. "
            "Monitor for complications: pneumothorax, hemothorax. CT for suspected complications. "
            "Multiple fractures or flail chest may require hospitalization. "
            "Evaluate for underlying cause if pathologic fracture suspected."
        ),
    },
    # ── MASS / TUMOR ──
    {
        "topic": "Lung Mass and Malignancy",
        "content": (
            "Lung mass is a pulmonary opacity >3cm. High suspicion for malignancy, especially in smokers >50 years. "
            "Types: non-small cell lung cancer (85%: adenocarcinoma, squamous cell, large cell) and "
            "small cell lung cancer (15%). Imaging features: spiculated borders, upper lobe predominance, "
            "associated lymphadenopathy, bone destruction, pleural effusion. "
            "Staging uses TNM classification. Metastatic disease may present as multiple bilateral nodules."
        ),
        "severity": "High",
        "recommendation": (
            "Urgent CT chest with contrast for characterization and staging. "
            "PET/CT for metabolic assessment. Tissue biopsy (CT-guided, bronchoscopy, or surgical). "
            "Multidisciplinary tumor board discussion. Stage-appropriate treatment: "
            "surgery, chemotherapy, radiation, immunotherapy, or targeted therapy."
        ),
    },
    {
        "topic": "Pulmonary Metastasis",
        "content": (
            "Pulmonary metastases appear as multiple bilateral round nodules of varying sizes "
            "(cannonball metastases). Common primary sites: breast, colon, kidney, melanoma, sarcoma. "
            "Lymphangitic carcinomatosis shows reticular pattern with septal thickening and pleural effusion. "
            "Endobronchial metastasis may cause post-obstructive pneumonia."
        ),
        "severity": "High",
        "recommendation": (
            "Identify primary malignancy if unknown. CT chest/abdomen/pelvis for full staging. "
            "PET/CT for metabolic activity. Biopsy of accessible lesion. "
            "Oncology referral for systemic therapy discussion."
        ),
    },
    # ── VASCULAR ──
    {
        "topic": "Pulmonary Embolism",
        "content": (
            "Pulmonary embolism (PE) is occlusion of pulmonary arteries by thrombus. "
            "Chest X-ray is often normal or shows nonspecific findings: atelectasis, small effusion, "
            "elevated hemidiaphragm. Classic signs (rare): Hampton's hump (wedge-shaped opacity), "
            "Westermark's sign (focal oligemia), Fleischner's sign (enlarged central pulmonary artery). "
            "CT pulmonary angiography (CTPA) is the diagnostic standard."
        ),
        "severity": "High",
        "recommendation": (
            "If PE suspected, obtain D-dimer (low-risk) or CTPA directly (high-risk). "
            "Start anticoagulation immediately if high clinical suspicion. "
            "Massive PE with hemodynamic instability: consider thrombolysis or embolectomy. "
            "Risk stratify using PESI or sPESI score."
        ),
    },
    {
        "topic": "Aortic Dissection",
        "content": (
            "Aortic dissection is a tear in the aortic intima allowing blood to enter the media. "
            "Stanford Type A: involves ascending aorta (surgical emergency). "
            "Stanford Type B: involves descending aorta distal to left subclavian. "
            "Chest X-ray findings: widened mediastinum, irregular aortic contour, left pleural effusion. "
            "CTA is the diagnostic standard showing intimal flap and true/false lumen."
        ),
        "severity": "High",
        "recommendation": (
            "Type A: emergency surgical repair. Type B: medical management with blood pressure control "
            "(target SBP <120mmHg). IV beta-blockers first line. Urgent CTA for diagnosis. "
            "Cardiothoracic surgery consultation. Monitor for complications: "
            "malperfusion, aortic rupture, organ ischemia."
        ),
    },
    # ── DEGENERATIVE / CHRONIC ──
    {
        "topic": "Degenerative Spine Disease",
        "content": (
            "Degenerative changes in the spine include disc space narrowing, osteophyte formation, "
            "facet joint arthropathy, and endplate sclerosis. Spondylosis refers to age-related degeneration. "
            "Disc herniation can compress nerve roots causing radiculopathy. "
            "Spinal stenosis causes neurogenic claudication. "
            "Most common in cervical (C5-C7) and lumbar (L4-S1) regions."
        ),
        "severity": "Low to Medium",
        "recommendation": (
            "Conservative management: physical therapy, NSAIDs, weight management. "
            "MRI if neurological symptoms (weakness, numbness, bowel/bladder changes). "
            "Referral to spine specialist if refractory to conservative treatment. "
            "Red flags: progressive neurological deficit, cauda equina syndrome."
        ),
    },
    {
        "topic": "Scoliosis",
        "content": (
            "Scoliosis is lateral curvature of the spine >10 degrees (Cobb angle). "
            "Types: idiopathic (most common, adolescent onset), congenital, neuromuscular, degenerative. "
            "Mild: 10-25 degrees. Moderate: 25-40 degrees. Severe: >40 degrees. "
            "Assess for rotational deformity and rib hump. "
            "Progressive curves may affect pulmonary function in severe cases."
        ),
        "severity": "Low to Medium",
        "recommendation": (
            "Mild: observation and monitoring, especially in growing children. "
            "Moderate: bracing in skeletally immature patients. "
            "Severe or progressive: surgical consultation for spinal fusion. "
            "Monitor pulmonary function in severe thoracic curves."
        ),
    },
    # ── NORMAL FINDINGS ──
    {
        "topic": "Normal Chest X-ray Findings",
        "content": (
            "A normal chest X-ray shows clear lung fields without focal consolidation, mass, or effusion. "
            "Normal cardiac silhouette with cardiothoracic ratio ≤0.5 on PA view. "
            "Normal mediastinal contours. Clear costophrenic angles. "
            "Intact bony thorax without fracture or lytic lesion. "
            "Normal soft tissues. Trachea midline. Normal aortic knob."
        ),
        "severity": "Low",
        "recommendation": (
            "No acute intervention required. Routine follow-up as clinically indicated. "
            "If symptoms persist despite normal imaging, consider advanced imaging (CT) "
            "or alternative diagnoses."
        ),
    },
    {
        "topic": "Atelectasis",
        "content": (
            "Atelectasis is partial or complete collapse of lung parenchyma. "
            "Types: obstructive (mucus plug, tumor), compressive (effusion, pneumothorax), "
            "passive (loss of contact between pleura), adhesive (surfactant deficiency). "
            "X-ray findings: increased opacity, volume loss, shift of fissures, elevated hemidiaphragm, "
            "mediastinal shift toward atelectasis. Subsegmental atelectasis shows linear opacities (plate-like)."
        ),
        "severity": "Low to Medium",
        "recommendation": (
            "Treat underlying cause. Incentive spirometry and chest physiotherapy. "
            "Bronchoscopy if obstructive cause suspected. "
            "Post-surgical atelectasis usually resolves with mobilization and deep breathing exercises."
        ),
    },
    # ── ADDITIONAL KNOWLEDGE ──
    {
        "topic": "Calcification in Chest Imaging",
        "content": (
            "Calcifications in the chest can be benign or pathologic. "
            "Benign patterns: central, popcorn (hamartoma), laminar, diffuse. "
            "Coronary artery calcification indicates atherosclerosis. "
            "Mediastinal lymph node calcification: prior granulomatous infection (TB, histoplasmosis). "
            "Pleural calcification: prior empyema, hemothorax, or asbestos exposure. "
            "Pericardial calcification: constrictive pericarditis."
        ),
        "severity": "Low",
        "recommendation": (
            "Most calcifications are incidental and benign. Compare with prior imaging. "
            "Coronary calcification: cardiac risk assessment. "
            "Pleural calcification with asbestos history: screen for mesothelioma. "
            "Document pattern and location for future comparison."
        ),
    },
    {
        "topic": "Mediastinal Widening",
        "content": (
            "Mediastinal widening (>8cm on PA chest X-ray) can indicate serious pathology. "
            "Differential: aortic dissection/aneurysm, lymphadenopathy (lymphoma, sarcoidosis, metastatic), "
            "mediastinal mass (thymoma, teratoma, thyroid goiter), mediastinal hemorrhage (trauma), "
            "and normal variant (obesity, AP projection artifact). "
            "CT with contrast is the standard for further evaluation."
        ),
        "severity": "Medium to High",
        "recommendation": (
            "Urgent CT chest with contrast to determine cause. If aortic dissection suspected, "
            "CTA is emergent. For lymphadenopathy, consider tissue sampling. "
            "For anterior mediastinal mass, evaluate with MRI and tumor markers."
        ),
    },
    {
        "topic": "Interstitial Lung Disease",
        "content": (
            "Interstitial lung disease (ILD) encompasses a group of disorders causing diffuse parenchymal disease. "
            "Chest X-ray patterns: reticular (lines), nodular, reticulonodular, ground-glass, honeycombing. "
            "Common types: idiopathic pulmonary fibrosis (IPF), sarcoidosis, hypersensitivity pneumonitis, "
            "connective tissue disease-associated ILD, drug-induced ILD. "
            "Distribution helps narrow diagnosis: upper-lobe predominant (sarcoidosis, silicosis), "
            "lower-lobe predominant (IPF, asbestosis)."
        ),
        "severity": "Medium to High",
        "recommendation": (
            "High-resolution CT (HRCT) is essential for pattern recognition. "
            "Pulmonary function tests (restrictive pattern with decreased DLCO). "
            "Rheumatologic workup if connective tissue disease suspected. "
            "Multidisciplinary discussion for ILD classification. "
            "Consider lung biopsy if diagnosis uncertain."
        ),
    },
]


def get_all_documents():
    """Return all knowledge base documents as a list of strings for embedding."""
    return [
        f"{doc['topic']}: {doc['content']} Severity: {doc['severity']}. Recommendation: {doc['recommendation']}"
        for doc in MEDICAL_KNOWLEDGE
    ]


def get_all_metadata():
    """Return metadata for each document."""
    return [
        {
            "topic": doc["topic"],
            "severity": doc["severity"],
            "recommendation": doc["recommendation"],
        }
        for doc in MEDICAL_KNOWLEDGE
    ]
