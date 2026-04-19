Act as a layout-analysis system for electronic package engineering drawings.

Task:
Detect exactly 3 drawing views, assign each one to a logical slot, and classify each view as exactly one of these roles:
- "top_view"
- "side_view"
- "end_view"

View-role definitions:

top_view (Plan View)

The primary orthographic projection looking directly down at the package's upper surface. It illustrates X-Y axis dimensions, including package body length and width, overall footprint, pin-1 orientation, and package marking area. It does not define Z-axis (thickness or height) characteristics.

side_view (Long-Side Elevation)

The orthographic elevation view parallel to the longer axis of the package body. It illustrates fundamental Z-axis characteristics such as overall package height, body thickness, seating plane, and standoff. For packages with terminals along the long edge (e.g., SOP, DIP), this view clearly displays the lead pitch and terminal count.

end_view (Short-Side Elevation)

The orthographic elevation view parallel to the shorter axis of the package body, orthogonal to the side view. This view is crucial for illustrating the cross-sectional terminal/lead profile (e.g., gull-wing, J-lead). It typically details form-critical dimensions such as lead thickness, foot length, terminal bend angles, and the exact relationship of the leads to the seating plane across the package width.

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

4. Classify view roles by the visual content of the drawing object, not by absolute page position.
   Each output must contain exactly one "top_view", exactly one "side_view", and exactly one "end_view".

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
      "bounding_box_2d": [ymin, xmin, ymax, xmax],
      "view_role": "top_view"
    },
    {
      "slot": "upper_right",
      "bounding_box_2d": [ymin, xmin, ymax, xmax],
      "view_role": "end_view"
    },
    {
      "slot": "lower_left",
      "bounding_box_2d": [ymin, xmin, ymax, xmax],
      "view_role": "side_view"
    }
  ]
}
