Act as an OCR and layout-analysis system for engineering drawings.

Task:
Detect exactly 3 drawing objects, assign each one to a logical slot, and extract all horizontal and vertical dimension values as strict JSON.

Rules:
1. Do not name views. Use only these logical slots:
   - "upper_left"
   - "upper_right"
   - "lower_left"
   - "lower_right"

2. Do not split the page into four geometric quadrants.
   First detect the 3 drawing objects, estimate one bounding box for each, and assign each object to a slot using the relative positions of the object centers.
   Exactly 1 slot is empty.

3. A drawing object may visually extend beyond its expected area.
   Slot assignment must be based on the object's relative center position among the 3 detected objects, not on page overlap.

4. Extract only dimensions whose measured direction is:
   - "horizontal"
   - "vertical"

5. Read numeric dimension text exactly as printed. Decimal points are critical.
   - Preserve a leading decimal point when the printed value begins with ".".
   - Example: output ".1" when the drawing shows ".1"; output ".22" when the drawing shows ".22".
   - Do not add a leading zero unless the zero is visibly printed in the drawing.
   - Never drop a visible decimal point.
   - For any dimension candidate that appears to be a single digit ("1" to "9"), perform a second check for a tiny leading decimal point.
   - If a leading decimal point is present, output ".x", not "x".

6. Assign each dimension to the drawing object it annotates, not to the page area where the text appears.
   A dimension label or extension line may extend outside the main visual area of its object and still belongs to that object.

7. Bounding boxes must use normalized coordinates:
   [ymin, xmin, ymax, xmax]
   Scale: 0-1000

8. Return only valid JSON. No markdown. No explanation.
  - Respond ONLY with a valid JSON object.
  - Do NOT include Markdown code fences.
  - Do NOT start the response with ```json, ```, or any other Markdown wrapper.
  - Do NOT provide explanations, notes, or conversational text before or after the JSON.
  - The first character of the response must be "{"
  - The last character of the response must be "}"

Use this exact JSON structure:

{
  "layout": {
    "upper_left": 1,
    "upper_right": 1,
    "lower_left": 1,
    "lower_right": 0
  },
  "views": [
    {
      "slot": "upper_left",
      "bounding_box_2d": [ymin, xmin, ymax, xmax],
      "dimensions": [
        {
          "value": ".1",
          "orientation": "vertical",
          "belongs_to_slot": "upper_left"
        }
      ]
    }
  ]
}
