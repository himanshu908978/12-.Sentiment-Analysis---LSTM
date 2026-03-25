const uploadimg = document.querySelector(".uploadimg");
const uploadimgc = document.querySelector(".uploadimgc");
const input1 = document.querySelector("#input1");
const img1 = document.querySelector(".img1");
const uploadbtn = document.querySelector(".uploadbtn");
const firstb1 = document.querySelector(".firstb1");
const results = document.querySelector(".results");

results.addEventListener("click",()=>{
    everythingcontainer1.style.display = "block";
})

firstb1.addEventListener("click",()=>{
    Everything.style.display = "flex";
})



// input1.addEventListener("change",uploadimginp);
// function uploadimginp(){
//     let imglink = URL.createObjectURL(input1.files[0]);
//     img1.src = `${imglink}`;
//     // uploadimg.style.backgroundImage = `url(${imglink})`;
//     img1.style.display = "block";
//     // uploadimg.style.display = "none";
//     uploadimgc.style.display = "none";
//     uploadimg.style.backgroundColor= "rgb(228, 228, 228)";
// }


uploadbtn.addEventListener("click",async()=>{
    try{const input = document.querySelector(".sentinput").value;
    if(!input) return;
    uploadbtn.innerText = "Predicting...";

    const result = await fetch("http://127.0.0.1:8000/sentiment",{
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body:JSON.stringify({input_text:input})
    });

    const data = await result.json();
    const answer = document.querySelector(".answer");
    const answer1 = document.querySelector(".answer1");

    answer.style.display = "block";
    answer1.style.display = "block";

    answer.innerText = `The Response in sentiment is ${data.pred_class}.`
    answer1.innerText = `Model confidence score is ${data.conf.toFixed(3)}.`
    uploadbtn.innerText = "Upload Sentiment";
    }
    catch(error){
        alert("Internal server error.");
        return;
    }
})





const projectname = document.querySelector(".projectname");
const Everything = document.querySelector(".Everything");

projectname.addEventListener("click",()=>{
    Everything.style.display = "none";
});


const projectname1 = document.querySelector(".projectname1");
const everythingcontainer1 = document.querySelector(".everythingcontainer1");

projectname1.addEventListener("click",()=>{
    everythingcontainer1.style.display = "none";
})


const textarea = document.querySelector(".sentinput");

const MIN_HEIGHT = 5;   
const MAX_HEIGHT = 30;  
function adjustHeight() {

    textarea.style.height = "auto";

    let newHeight = textarea.scrollHeight;

    const maxHeightPx =
        window.innerHeight * (MAX_HEIGHT / 100);

    if (newHeight > maxHeightPx) {

        textarea.style.height = maxHeightPx + "px";
        textarea.style.overflowY = "auto";

    } else {

        textarea.style.height = newHeight + "px";
        textarea.style.overflowY = "hidden";

    }
}

textarea.addEventListener("input", adjustHeight);