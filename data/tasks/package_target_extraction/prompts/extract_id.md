Based on the image, identify the package dimensions according to the following size-based definitions.

{{PACKAGE_CONTEXT_BLOCK}}

1. Dimension system

Use these three output fields as three distinct package-dimension concepts:

1. `body_side_dimensions`
2. `maximum_terminal_to_terminal_span`
3. `overall_package_height`

2. Measurement scope

Use two planar measurement scopes and one perpendicular measurement scope.

**Body-only planar dimensions**
These are the two orthogonal package-body dimensions measured parallel to the seating plane, on the package body only, and excluding terminals.

- `body_side_dimensions`: return the two body-only planar dimensions together, without assigning which one is longer or shorter.

**Terminal-including planar overall dimension**
This is measured parallel to the seating plane and includes terminals.

- `maximum_terminal_to_terminal_span`: for packages with terminals on two opposite sides, the largest overall planar dimension measured from the outermost terminal tip on one side to the outermost terminal tip on the opposite side.

**Perpendicular overall package dimension**
This is measured perpendicular to the seating plane.

- `overall_package_height`: the total packaged height, measured from the seating plane to the highest point of the component.

3. Orientation basis

- "Parallel to the seating plane" means dimensions lying in the seating-plane directions.
- "Perpendicular to the seating plane" means the height direction normal to the seating plane.

4. Identifier extraction rule

Return the drawing identifier for each required dimension concept, not the numeric measurement value itself.

- Each returned value must be an identifier in the exact form `"ID{number}"`.
- For `body_side_dimensions`, return exactly two identifier strings when both body-only planar dimensions can be identified.
- Do not assign long-side or short-side semantics inside `body_side_dimensions`.
- Do not use an identifier for a body-only dimension as a substitute for `maximum_terminal_to_terminal_span`.
- If the package does not have terminals on two opposite sides, or if the terminal-to-terminal span cannot be determined, return `null` for `maximum_terminal_to_terminal_span`.
- If one or both body-only planar dimensions cannot be determined, return `null` for `body_side_dimensions`.
- If the overall package height cannot be determined, return `null` for `overall_package_height`.

{{VIEW_SEMANTICS_WARNING}}

Return valid JSON only using this exact schema:
{
  "body_side_dimensions": ["ID{number}", "ID{number}"] | null,
  "maximum_terminal_to_terminal_span": "ID{number}" | null,
  "overall_package_height": "ID{number}" | null
}

Rules:
- Return only identifier strings in the exact format `"ID{number}"`, or `null`.
- Do not return the numeric measurement values themselves.
- Do not include units, dimension letters, JEDEC symbols, confidence, comments, or explanations.
- Respond ONLY with a valid JSON object.
- Do NOT include Markdown code fences.
- Do NOT start the response with ```json, ```, or any other Markdown wrapper.
- Do NOT provide explanations, notes, or conversational text before or after the JSON.
- The first character of the response must be "{"
- The last character of the response must be "}"