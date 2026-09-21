# 🧪 Testing Guide: Advanced RAG with Hybrid Dataset

This guide helps you systematically test the coffee disease diagnosis system with hybrid JSON+PDF retrieval.

---

## 📊 System Features (Quick Reference)

| Component | Implementation |
|-----------|---------------|
| **Dataset** | Hybrid: 60% JSON (15 diseases) + 40% PDF (supporting evidence) |
| **Query Processing** | 3 query variants (base, context, keywords) |
| **Scoring** | Hybrid (vector 55% + lexical 35% + variant 10%) |
| **Diversity** | Max 2 chunks per source |
| **Deduplication** | Content-based (220 char prefix) |
| **Metadata Filtering** | Filter by symptom type, region, affected parts |
| **Ambiguity Handling** | Strict - always asks on vague input |
| **Diagnosis** | Structured candidate comparison |
| **Verification** | Active gating (can request more info) |
| **Question Length** | Concise (20-25 words, no double ??) |

---

## 🎯 Test Categories

### 1️⃣ **HYBRID RETRIEVAL BALANCE TEST** (Most Important)
**Goal**: Verify balanced disease distribution (no Coffee Leaf Rust bias)

#### Automated Test
```bash
python test/test_hybrid.py
```

**What it Tests**:
- 8 diverse symptom queries
- Disease distribution across diagnoses
- JSON vs PDF source ratio (target: 60/40)
- Source diversity enforcement

**Success Criteria**:
- ✅ Coffee Leaf Rust should be <70% of diagnoses
- ✅ Diverse diseases for diverse symptoms
- ✅ Each query shows mix of [JSON] and [PDF] sources
- ✅ No single disease dominates

#### Manual Test Cases

**Test Case 1.1: Nutrient Deficiency**
```
Initial Input: "Leaves turning yellow uniformly"

Expected:
- Should retrieve Nitrogen Deficiency or Magnesium Deficiency (from JSON)
- Should ask about vein pattern (stay green or also yellow?)
- Should show [JSON] tags for primary disease info
- Should NOT immediately diagnose Coffee Leaf Rust
```

**Test Case 1.2: Berry Problems**
```
Initial Input: "Small holes in coffee berries"

Expected:
- Should diagnose Coffee Berry Borer (from JSON)
- Should retrieve pest-related information
- Should show [JSON] source with symptom_type='pest_berry'
```

**Test Case 1.3: Root/Wilting Issues**
```
Initial Input: "Plant is wilting despite adequate watering"

Expected:
- Should consider Root Rot or Coffee Wilt Disease
- Should ask about root color, soil drainage
- Should NOT default to foliar diseases
```

---

### 2️⃣ **SOURCE DIVERSITY & TAGGING TEST**
**Goal**: Verify chunks come from both JSON and PDF sources with proper tags

#### Test Case 2.1: Check Source Tags in UI
```
Initial Input: "Yellowing between leaf veins"

What to Check in UI:
1. Evidence section should show source tags: [JSON] or [PDF]
2. Should see mix of both source types
3. [JSON] chunks should show structured disease info
4. [PDF] chunks should show supporting details
```

**How to Check**:
- Run Streamlit UI: `python -m streamlit run ui/streamlit_app.py`
- Enter query and proceed to diagnosis
- Scroll to "Evidence (RAG sources)" section
- Verify tags are displayed: **[JSON] disease_knowledge.json** or **[PDF] Coffee (1).pdf**

---

### 3️⃣ **AMBIGUITY STRICTNESS TEST**
**Goal**: System should ask questions for vague inputs (not guess)

#### Test Case 3.1: Very Vague Initial Input
```
Initial Input: "Something wrong with leaves"

Expected (Advanced):
- MUST ask clarifying question (color? pattern? location?)
- Should NOT attempt diagnosis

Expected (Naive):
- Might proceed to diagnosis with limited info
```

#### Test Case 3.2: Missing Critical Details
```
Initial Input: "Leaves have spots"

Expected (Advanced):
- Should ask about spot color, size, pattern
- Should ask about leaf location
- Should need 2-3 follow-ups before diagnosis

Expected (Naive):
- Might diagnose after 1 question
```

---

### 4️⃣ **ACTIVE GATING TEST**
**Goal**: System requests more info when verification is weak

#### Test Case 4.1: Borderline Symptoms
```
Initial Input: "Leaves yellowing with small spots"
Follow-up 1: "On lower leaves"
Follow-up 2: "Spots are brownish"

Expected (Advanced):
- If diagnosis consistency is low, system asks ONE more question
- Terminal shows: "Verification weak - requesting additional clarification"

Expected (Naive):
- Proceeds to diagnosis even if inconsistent
```

**How to Observe**:
- Check terminal for verification scores
- Look for: "Consistency: 66.7%" or similar
- Advanced should ask another question if < 80% consistency

---

### 5️⃣ **MULTI-QUERY EFFECTIVENESS TEST**
**Goal**: Same symptom, different phrasing should work

#### Test Case 5.1: Synonym Variations
```
Test A: "Leaves are yellowing"
Test B: "Foliage turning yellow"
Test C: "Chlorosis on leaves"

Expected (Advanced):
- All three should retrieve similar relevant chunks
- Multi-query expansion captures semantic variations

Expected (Naive):
- Exact keyword matching might miss variations
```

---

### 6️⃣ **STRUCTURED DIAGNOSIS TEST**
**Goal**: Verify differential diagnosis (candidate comparison)

#### Test Case 6.1: Ambiguous Symptoms
```
Input: "Yellow leaves with brown edges on lower branches"

Expected (Advanced):
- Terminal/log shows candidate comparison:
  - Coffee Leaf Rust: 60% (evidence: ...)
  - Nitrogen Deficiency: 75% (evidence: ...)
  - Leaf Blight: 45% (evidence: ...)
- Final diagnosis picks highest score

Expected (Naive):
- Direct output without comparison
- May default to most common disease
```

**How to Check**:
- Look for log line: "Candidate comparison:"
- Check if multiple diseases are evaluated

---

## 🔬 Step-by-Step Testing Procedure

### Preparation
1. **Ensure Ollama is running**:
   ```bash
   ollama list
   ```
   Verify your model (e.g., `phi3`) is available

2. **Enable Debug Logging**:
   Edit `config/settings.py`:
   ```python
   LOG_LEVEL=DEBUG
   ```

3. **Clear Previous State** (optional):
   ```bash
   rm -rf vector_db/
   ```
   This rebuilds embeddings (takes 2-3 minutes)

### Running Tests

#### Option 1: Streamlit UI (Visual)
```bash
python -m streamlit run ui/streamlit_app.py
```
- Test cases one by one
- Observe question flow
- Check final diagnosis
- Take screenshots for comparison

#### Option 2: CLI Testing (Faster)
```bash
python test/test_diagnosis.py "Leaves are turning yellow"
```
- Observe terminal output
- Check retrieved sources
- Note question quality

#### Option 3: Automated Hybrid Test
```bash
python test/test_hybrid.py
```
- Tests 8 diverse symptom scenarios
- Measures disease distribution
- Validates JSON vs PDF ratio
- Checks for bias (Coffee Leaf Rust <70%)

#### Option 4: Batch Testing (Most Thorough)
Create a test script:

```python
# custom_test_suite.py
from src.coffee_diagnosis.diagnosis.controller import CoffeeDiagnosisController

test_cases = [
    "Leaves turning yellow uniformly",
    "Dark spots with orange powder on leaves",
    "Plant wilting despite watering",
    "Brown edges on older leaves",
    "Small holes in coffee berries"
]

controller = CoffeeDiagnosisController(data_dir="data", vector_db_path="data/vector_db")

for test in test_cases:
    print(f"\n{'='*60}")
    print(f"TEST: {test}")
    print('='*60)
    result = controller.start_diagnosis(test)
    
    # Skip clarification for quick test
    if result['status'] == 'question':
        result = controller.continue_with_answer("skip to diagnosis")
    
    if result['status'] == 'diagnosis':
        diagnosis = result['diagnosis']
        print(f"Diagnosis: {diagnosis['disease']}")
        print(f"Confidence: {diagnosis.get('confidence', 'N/A')}")
        
        # Check sources
        ctx = controller.state_manager.state.retrieved_context
        json_count = sum(1 for c in ctx if c.get('source_type') == 'JSON')
        pdf_count = sum(1 for c in ctx if c.get('source_type') == 'PDF')
        print(f"Sources: {json_count} JSON, {pdf_count} PDF")
    
    controller.reset()
```

---

## 📈 Metrics to Track

### Quantitative Metrics

| Metric | How to Measure | Target |
|--------|----------------|--------|
| **Disease Distribution** | Run `python test/test_hybrid.py` | No disease >70% |
| **JSON/PDF Ratio** | Check UI source tags | ~60% JSON, ~40% PDF |
| **Source Diversity** | Count unique sources in evidence | ≥ 2 sources |
| **Question Quality** | Manually review questions | Concise (20-25 words), no ?? |
| **Diagnosis Accuracy** | Manual validation against symptoms | > 80% |
| **Consistency Score** | Verification report in UI | > 80% |
| **Questions Before Diagnosis** | Count in conversation | 1-3 for vague input |

### Qualitative Observations

**For Each Test Case, Note**:
1. ✅ Did system show [JSON] and [PDF] tags in evidence?
2. ✅ Were retrieved chunks from multiple diseases (not just rust)?
3. ✅ Did diagnosis match symptom details accurately?
4. ✅ Was there candidate comparison (multiple diseases evaluated)?
5. ✅ Did active gating work (asked more if uncertain)?
6. ✅ Were questions concise and clear (no double ??)?

---

## 🐛 Common Issues vs Expected Behavior

### Issue 1: Still Getting Coffee Leaf Rust Always
**Check**:
- Is `disease_knowledge.json` present in `data/` folder?
- Are 15 diseases loaded? (Check console on startup: "Loaded 15 diseases from JSON")
- Is JSON retriever initialized? (Should see "Loading structured JSON disease knowledge...")
- Delete `data/vector_db/` and restart to rebuild indexes

### Issue 2: No [JSON] Tags Showing
**Check**:
- Verify `disease_knowledge.json` exists at `data/disease_knowledge.json`
- Check console for JSON loading errors
- Ensure controller initialization succeeded

### Issue 3: Too Many Questions
**This is OK!** The system is strict about ambiguity.
- If symptoms are vague, 2-3 questions is expected
- Better than wrong diagnosis
- Questions should be concise (20-25 words)

### Issue 4: Different Diagnosis Than Before
**This is GOOD!** The hybrid system has balanced knowledge.
- Compare with JSON disease definitions
- Should be more accurate with diverse disease coverage

---

## 📝 Test Report Template

After testing, document results:

```markdown
## Test Report: [Date]

### Test Case: [Name]
**Input**: "[symptom description]"

**System Behavior**:
- Questions asked: X
- Question quality: [Concise? No double ??]
- Sources retrieved: 
  - JSON: X chunks from disease_knowledge.json
  - PDF: X chunks from [PDF names]
- Source tags visible: Yes/No
- Candidates compared: [list diseases from terminal logs]
- Final diagnosis: [disease name]
- Confidence: XX%
- Verification score: XX%
- Active gating triggered: Yes/No

**Disease Knowledge Source**:
- Primary info from: [JSON/PDF]
- Supporting info from: [JSON/PDF]

**Observations**:
- ✅ Balanced retrieval (no rust bias)?
- ✅ Accurate diagnosis?
- ✅ Good question quality?
- ❌ Any issues noted?
```

---

## 🎯 Expected System Behavior Summary

After testing, you should observe:

1. **✅ Balanced Disease Coverage**: 
   - 15 diseases available in JSON
   - No single disease dominates diagnoses
   - Coffee Leaf Rust <70% across diverse queries

2. **✅ Hybrid Source Retrieval**: 
   - ~60% chunks from [JSON] disease_knowledge.json
   - ~40% chunks from [PDF] files
   - Both source types visible with tags

3. **✅ Metadata Filtering**: 
   - JSON retrieval filtered by symptom type when applicable
   - Faster, more targeted retrieval

4. **✅ Concise Questions**: 
   - 20-25 words per question
   - No double question marks (??)
   - Clear, direct language

5. **✅ Stricter Ambiguity Handling**: 
   - Asks questions for vague inputs
   - Doesn't guess when uncertain

6. **✅ Candidate Comparison**: 
   - Multiple diseases evaluated before final pick
   - Structured diagnosis output

7. **✅ Active Gating**: 
   - Requests more info when verification weak (<80%)
   - Shows "Verification Alert" notification in UI

---

## 🚀 Quick Smoke Test (5 Minutes)

Run these tests to quickly validate the hybrid system:

```bash
# Test 1: Automated balance check
python test/test_hybrid.py
# Expected: Coffee Leaf Rust <70%, diverse diagnoses

# Test 2: UI source tags
python -m streamlit run ui/streamlit_app.py
# Input: "Leaves yellowing between veins"
# Expected: See [JSON] and [PDF] tags in evidence section

# Test 3: Specific symptom
# Input: "Yellow leaves with orange pustules on undersides"
# Expected: Diagnoses Coffee Leaf Rust (correct), shows both JSON and PDF sources
```

---

**Good luck testing! 🎉** The hybrid system should show clear improvements in balanced disease coverage and source transparency.

