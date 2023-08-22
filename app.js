const form = document.querySelector('#input-form');
const query = document.querySelector('#query');
const wordcount = document.querySelector('#wordcount');
const urls = document.querySelector('#urls');
const summary = document.querySelector('#summary');

form.addEventListener('submit', (e) => {
	e.preventDefault();
	fetch('/summarize', {
		method: 'POST'
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({
			query: query.value,
			wordcount: wordcount.value,
			urls: urls.value
		})
	})
	.then(response => response.json())
	.then(data => {
		summary.innerHTML = data.summary;
	})
	.catch(error => console.error(error));
});
