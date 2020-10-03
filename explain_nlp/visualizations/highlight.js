function visualizeExample(sequence, label, importances) {
	// TODO: display the actual importances somewhere (make sure to round them to a few decimals though)
	let examples_div = document.getElementsByClassName("examples")[0];

	let minImportance = importances.reduce((currMin, currValue) => currValue < currMin? currValue: currMin);
	let maxImportance = importances.reduce((currMax, currValue) => currValue > currMax? currValue: currMax);

	// scale importances to [-1, 1] range
	let scaledImportances = importances.map(imp => (1 - (-1)) * (imp - minImportance) / (maxImportance - minImportance) + (-1));

	let createdSpans = scaledImportances.map((imp, i) => {
		if(imp < 0) {
			return `<span style="background-color: rgba(255, 0, 0, ${Math.abs(imp)})">${sequence[i]}</span>`
		}
		else if (imp > 0) {
			return `<span style="background-color: rgba(0, 153, 0, ${imp})">${sequence[i]}</span>`
		}
		// leave 0 importance uncolored
		else
			return `<span>${sequence[i]}</span>`
	});
	examples_div.innerHTML += `<div class="example" style="word-break: break-word"><span class="example-label">Label: ${label}</span> <br />${createdSpans.join("")}</div>`;
}