function visualizeExample(sequence, label, importances, actualLabel) {
	// TODO: softmax the importances internally to ensure we can always use that number as the alpha channel in RGBA
	let examples_div = document.getElementsByClassName("examples")[0];

	let minImportance = importances.reduce((currMin, currValue) => currValue < currMin? currValue: currMin);
	let maxImportance = importances.reduce((currMax, currValue) => currValue > currMax? currValue: currMax);

	let maxAbsImportance = Math.max(Math.abs(minImportance), Math.abs(maxImportance));
	let bottomLimit = - maxAbsImportance;
	let topLimit = maxAbsImportance;

	// scale importances to [-maxAbsImportance, maxAbsImportance] range
	let scaledImportances = importances.map(imp => (1 - (-1)) * (imp - bottomLimit) / (topLimit - bottomLimit) + (-1));

	let createdSpans = scaledImportances.map((imp, i) => {
		let roundedOrigImportance = importances[i].toFixed(3);

		if(imp < 0) {
			return `<span title="${roundedOrigImportance}" style="background-color: rgba(255, 0, 0, ${Math.abs(imp)})">${sequence[i]}</span>`;
		}
		else if (imp > 0) {
			return `<span title="${roundedOrigImportance}" style="background-color: rgba(0, 153, 0, ${imp})">${sequence[i]}</span>`;
		}
		// leave 0 importance uncolored
		else
			return `<span title="${roundedOrigImportance}">${sequence[i]}</span>`;
	});

	let isGroundTruthGiven = actualLabel !== "";
	let isLabelCorrect = isGroundTruthGiven && (label === actualLabel);
	let predictedLabelSpan;
	if(!isGroundTruthGiven) {
		predictedLabelSpan = `<span class="example-label" title="Actual: /" style="background-color: rgb(238, 232, 170)">Predicted label: ${label}</span>`;
	}
	else {
		let color = isLabelCorrect? "rgba(200, 247, 197, 1)": "rgba(236, 100, 75, 1)";
		predictedLabelSpan = `<span class="example-label" title="Actual: ${actualLabel}" style="background-color: ${color}">Predicted label: ${label}</span>`;
	}

	examples_div.innerHTML += `<div class="example" style="word-break: break-word">${predictedLabelSpan}<br />${createdSpans.join("")}</div>`;
}