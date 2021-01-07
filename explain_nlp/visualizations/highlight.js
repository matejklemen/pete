
function toggleDisplay(objectId) {
	let currDisplayStyle = document.getElementById(objectId).style.display;
	console.log(currDisplayStyle);
	if(currDisplayStyle === "none")
		document.getElementById(objectId).style.display = "block";
	else
		document.getElementById(objectId).style.display = "none";
}

function toggleMarked(caller) {
	/* Function used to (un)mark all spans corresponding to the same feature as 'selected'.
		Args:
		`caller`: object that has the attributes "data-sequence" and "data-feature", indicating which feature of which
			sequence is to be (un)marked
	 */
	let idxSeq = caller.getAttribute("data-sequence");
	let idxFeature = caller.getAttribute("data-feature");

	let exampleDiv = document.getElementById(idxSeq);
	for(let child of exampleDiv.children) {
		if(child.getAttribute("data-feature") !== idxFeature)
			continue

		child.classList.contains("selected")? child.classList.remove("selected"): child.classList.add("selected");
	}
}