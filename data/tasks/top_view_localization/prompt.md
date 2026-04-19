Act as a layout-analysis system for electronic package engineering drawings.

Task:
Detect exactly 3 drawing views, assign each one to a logical slot, and identify which slot contains the package top view.

Top-view definition:
In electronic package drawings, the package top view is the drawing object whose primary reference face is the package top surface, defined by package meaning rather than drawing direction. The package top surface is the identification face of the finished package: the face used for external package marking and orientation reference, typically associated with the body outline and pin-1 indicator or other package marking features. The opposite face is the mounting/connection face, which carries the leads, pads, lands, balls, or other electrical/thermal contact structures used to attach the package to the PCB. Therefore, the package top view is the drawing object corresponding to the package identification face, not the drawing object whose primary purpose is to define package height, seating plane, lead thickness, or other thickness-related side profile characteristics.

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

4. Identify the top view by the visual content of the drawing object, not by absolute page position.
   The top view may appear in different slots in canonical and rotated variants.

5. Bounding boxes must use normalized coordinates:
   [ymin, xmin, ymax, xmax]
   Scale: 0-1000

6. Return only valid JSON. No markdown. No explanation.
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
      "bounding_box_2d": [ymin, xmin, ymax, xmax]
    },
    {
      "slot": "upper_right",
      "bounding_box_2d": [ymin, xmin, ymax, xmax]
    },
    {
      "slot": "lower_left",
      "bounding_box_2d": [ymin, xmin, ymax, xmax]
    }
  ],
  "top_view_slot": "upper_left"
}
