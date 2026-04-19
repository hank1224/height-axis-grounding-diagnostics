Act as a spatial-axis analysis system for electronic package engineering drawings.

Task:
Detect exactly 3 drawing views from the provided image.

For each detected view, do the following:

1. Assign the view to one logical slot based on the relative center position of the detected drawing object.
   Allowed slots:
   - "upper_left"
   - "upper_right"
   - "lower_left"
   - "lower_right"

   Exactly 3 slots are occupied and exactly 1 slot is empty.

2. Estimate the 2D bounding box of the drawing object.
   Use normalized coordinates:
   [ymin, xmin, ymax, xmax]
   Scale: 0-1000

3. Determine the PCB mounting face direction for the component in that specific 2D view.

   Field name:
   "pcb_mounting_face_axis"

   Meaning:
   The signed drawing-space axis direction from the component/package body toward the PCB mounting/connection face.

   The PCB mounting/connection face is the side of the component that contacts or connects to the PCB through leads, pads, lands, tabs, balls, or thermal/electrical contact structures.

Allowed signed axes:
- "+X"
- "-X"
- "+Y"
- "-Y"
- "+Z"
- "-Z"

Drawing-space axis convention:
- Right on the 2D drawing is "+X".
- Left on the 2D drawing is "-X".
- Up on the 2D drawing is "+Y".
- Down on the 2D drawing is "-Y".
- Out of the drawing toward the viewer is "+Z".
- Into the drawing away from the viewer is "-Z".

Rules:
1. Do not name views. Use only the logical slots listed above.

2. Do not split the page into four geometric quadrants.
   First detect the 3 drawing objects, estimate one bounding box for each, and assign each object to a slot using the relative positions of the object centers.

3. A drawing object may visually extend beyond its expected area.
   Slot assignment must be based on the object's relative center position among the 3 detected objects, not on page overlap.

4. Determine "pcb_mounting_face_axis" from package geometry, lead/pad/contact structures, seating plane marks, and PCB mounting semantics.
  - Do not infer it from page position alone.
  - Do not assume the upper-left view is always the top view.
  - Do not assume a rotated drawing preserves the same axis direction as the canonical drawing.
  - If any required value cannot be determined from the image, return `null` for that field.

5. Return only valid JSON. No markdown. No explanation.
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
      "pcb_mounting_face_axis": "+X | -X | +Y | -Y | +Z | -Z | null"
    },
    {
      "slot": "upper_right",
      "bounding_box_2d": [ymin, xmin, ymax, xmax],
      "pcb_mounting_face_axis": "+X | -X | +Y | -Y | +Z | -Z | null"
    },
    {
      "slot": "lower_left",
      "bounding_box_2d": [ymin, xmin, ymax, xmax],
      "pcb_mounting_face_axis": "+X | -X | +Y | -Y | +Z | -Z | null"
    }
  ]
}
