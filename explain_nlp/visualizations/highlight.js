// TODO: define `selectedUnitIds` in header JS once using this
selectedUnitIds = [];
examples = document.getElementsByClassName("example");
for(let i = 0; i < examples.length; i++) {
	selectedUnitIds.push(null);
}

function selectExample(objectId) {
	// TODO: currently unused
	// call inside spans as selectExample(this.id)
	let idParts = objectId.split("-");
	let elementType = idParts[0];
	let row = parseInt(idParts[1]);
	let col = parseInt(idParts[2]);

	if(selectedUnitIds[row] !== null) {
		let currSelected = selectedUnitIds[row];
		currSelected.classList.remove("selected");
	}

	let newSelected = document.getElementById(objectId);
	newSelected.classList.add("selected");
	selectedUnitIds[row] = newSelected;
}