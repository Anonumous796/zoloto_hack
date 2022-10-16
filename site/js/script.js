/* Вurger menu */
const isMobile = {
    Android: function () {
        return navigator.userAgent.match(/Android/i);
    },

    BlackBerry: function () {
        return navigator.userAgent.match(/BlackBerry/i);
    },

    IOS: function () {
        return navigator.userAgent.match(/iPhone|iPad|iPod/i);
    },

    Opera: function () {
        return navigator.userAgent.match(/Opera Mini/i);
    },
    
    Windows: function () {
        return navigator.userAgent.match(/IEMobile/i);
    },

    any: function () {
        return (
        isMobile.Android() ||
        isMobile.BlackBerry() ||
        isMobile.IOS() ||
        isMobile.Opera() ||
        isMobile.Windows());
    }
};

if (isMobile.any()) {
    document.body.classList.add('_touch');

    let menuArrows = document.querySelectorAll('.menu_arrow');

    if (menuArrows.length > 0) {
        for (let index = 0; index < menuArrows.length; index++) {
            const menuArrow = menuArrows[index];
            menuArrow.addEventListener("click", function (e) {
                menuArrow.parentElement.classList.toggle('_active');
            });
        }
    }

    } else {
    document.body.classList.add ('_pc');
}

const iconMenu = document.querySelector('.menu_icon');
const menuBody = document.querySelector('.menu_body');

if (iconMenu) {
    iconMenu.addEventListener("click", function (e) {
        document.body.classList.toggle('_lock');
        iconMenu.classList.toggle('_active');
        menuBody.classList.toggle('_active');
    });
}

/* Smooth slide navigation */
const menuLinks = document.querySelectorAll('a[data-goto]');
if(menuLinks.length > 0) {
    menuLinks.forEach(menuLink => {
        menuLink.addEventListener("click", onMenuLinkClick);
    });

    
    function onMenuLinkClick(e) {
        const menuLink = e.target;
        if (menuLink.dataset.goto && document.querySelector(menuLink.dataset.goto)) {
            const gotoBlock = document.querySelector(menuLink.dataset.goto);
            const gotoBlockValue = gotoBlock.getBoundingClientRect().top + pageYOffset - document.querySelector('header').offsetHeight;

            if (iconMenu.classList.contains('_active')) {
                document.body.classList.remove('_lock');
                iconMenu.classList.remove('_active');
                menuBody.classList.remove('_active');
            }

            window.scrollTo({
                top:gotoBlockValue,
                behavior: "smooth"
            });
            e.preventDefault();
        }
    }
}



/* Form */
$('.button_form').on('click', function() {
    let $parts = $(this).closest('.wrapper3').find('.part');
    $parts.toggleClass('part_active');
});

  
/* Select */
let select = function () {
    let selectHeader = document.querySelectorAll('.select_header');
    let selectItem = document.querySelectorAll('.select_item');

    selectHeader.forEach(item => {
        item.addEventListener('click', selectToggle)
    });

    selectItem.forEach(item => {
        item.addEventListener('click', selectChoose)
    });

    function selectToggle() {
        this.parentElement.classList.toggle('is-active');
    }

    function selectChoose() {
        let text = this.innerText,
            select = this.closest('.select'),
            currentText = select.querySelector('.select_current');
        currentText.innerText = text;
        select.classList.remove('is-active');
    }
};
select();




/* Slider */
const swiper = new Swiper('.swiper', {
    // Optional parameters
    loop: false,
  
    // Navigation arrows
    navigation: {
      nextEl: '.swiper-button-next',
      prevEl: '.swiper-button-prev',
    },
    grabCursor: true,
    slideToClickedSlide: true,
  
    keyboard: {
        enabled: true,
        onlyInViewport: true,
        pageUpDown: true,
    },

    mousewhell: {
        sensitivity: 1,
    },
    /*slidesPerView: 10,*/
    delay: 5000,
    slidesPerGroup: 4,
    initialSlide: 0,
    spaceBetween: 50,
    slidesPerView: 6,
    
    breakpoints: {
        // when window width is <= 499px
        0: {
            slidesPerView: 3.5,

        },

        280: {
            slidesPerView: 3.5,
        },

        300: {
            slidesPerView: 4,
        },

        450: {
            slidesPerView: 4,
        },

        500: {
            slidesPerView: 5,
        },

        600: {
            slidesPerView: 5,
        },

        700: {
            slidesPerView: 5,
        },

        800: {
            slidesPerView: 5,
        },

        900: {
            slidesPerView: 6,
        },

        1000: {
            slidesPerView: 6,
        },

        1450: {
            slidesPerView: 9,
        },

        1500: {
            slidesPerView: 10,
        },
    }
  });
