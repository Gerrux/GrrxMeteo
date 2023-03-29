let active_btn_map_type = document.querySelector(".navbar-btn-t.active");
let active_btn_map_hour = document.querySelector(".navbar-btn-h.active");
let modeMap = "temperature";
let timeMap = "0h";
let currentMap;
document.querySelectorAll(".navbar-btn-t").forEach(function(btn) {
    btn.addEventListener("click", function() {
        active_btn_map_type.classList.remove("active");
        btn.classList.add("active");
        active_btn_map_type = btn;
        let colorbar = `#${modeMap}-colorbar`;
        document.querySelector(colorbar).classList.remove("active-colorbar");

        modeMap = btn.id;
        colorbar = `#${modeMap}-colorbar`;
        document.querySelector(colorbar).classList.add("active-colorbar");

        let id = modeMap + timeMap;
        currentMap.click();
        currentMap = document.getElementById(id).parentElement;
        currentMap.click();
    });
});

document.querySelectorAll(".navbar-btn-h").forEach(function(btn) {
    btn.addEventListener("click", function() {
        active_btn_map_hour.classList.remove("active");
        btn.classList.add("active");
        active_btn_map_hour = btn;

        timeMap = btn.id;
        let id = modeMap + timeMap;
        currentMap.click();
        currentMap = document.getElementById(id).parentElement;
        currentMap.click();
    });
});

function ready() {
    let id = modeMap + timeMap;
    currentMap = document.getElementById(id).parentElement;
    currentMap.click();
}

document.addEventListener("DOMContentLoaded", ready);
