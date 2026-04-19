Based on the image, identify the package dimensions according to the following size-based definitions.

{{PACKAGE_CONTEXT_BLOCK}}

1. Dimension system

Use these four output fields as four distinct package-dimension concepts:

1. `body_long_side`
2. `body_short_side`
3. `maximum_terminal_to_terminal_span`
4. `overall_package_height`

2. Measurement scope

Use two planar measurement scopes and one perpendicular measurement scope.

**Body-only planar dimensions**
These are measured on the package body only, parallel to the seating plane, and excluding terminals.

- `body_long_side`: the larger of the two orthogonal body-only planar dimensions.
- `body_short_side`: the smaller of the two orthogonal body-only planar dimensions.

**Terminal-including planar overall dimension**
This is measured parallel to the seating plane and includes terminals.

- `maximum_terminal_to_terminal_span`: for packages with terminals on two opposite sides, the largest overall planar dimension measured from the outermost terminal tip on one side to the outermost terminal tip on the opposite side.

**Perpendicular overall package dimension**
This is measured perpendicular to the seating plane.

- `overall_package_height`: the total packaged height, measured from the seating plane to the highest point of the component.

3. Orientation basis

- "Parallel to the seating plane" means dimensions lying in the seating-plane directions.
- "Perpendicular to the seating plane" means the height direction normal to the seating plane.

4. Extraction rule

For each output field, return the numeric value shown in the drawing for that exact dimension concept as defined above.

- Do not treat `body_long_side` or `body_short_side` as terminal-including dimensions.
- Do not treat `maximum_terminal_to_terminal_span` as a body-only dimension.
- If the package does not have terminals on two opposite sides, or if the terminal-to-terminal span cannot be determined from the image, return `null` for `maximum_terminal_to_terminal_span`.
- If any required value cannot be determined from the image, return `null` for that field.

{{VIEW_SEMANTICS_WARNING}}

Return valid JSON only using this exact schema:
{
  "body_long_side": number | null,
  "body_short_side": number | null,
  "maximum_terminal_to_terminal_span": number | null,
  "overall_package_height": number | null
}

Rules:
- Return only the numeric values from the drawing.
- Do not include units, dimension letters, JEDEC symbols, confidence, comments, or explanations.
- Respond ONLY with a valid JSON object.
- Do NOT include Markdown code fences.
- Do NOT start the response with ```json, ```, or any other Markdown wrapper.
- Do NOT provide explanations, notes, or conversational text before or after the JSON.
- The first character of the response must be "{"
- The last character of the response must be "}"