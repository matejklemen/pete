
function toggleDisplay(objectId) {
	let currDisplayStyle = document.getElementById(objectId).style.display;
	console.log(currDisplayStyle);
	if(currDisplayStyle === "none")
		document.getElementById(objectId).style.display = "block";
	else
		document.getElementById(objectId).style.display = "none";
}