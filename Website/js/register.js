document.getElementById('btn').addEventListener('click', function (event) {
    event.preventDefault();

    const uname = document.getElementById('uname').value.trim();
    const email = document.getElementById('email').value.trim();
    const psswd = document.getElementById('psswd').value.trim();
    const repsswd = document.getElementById('repsswd').value.trim();
    const pattern = /^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{6,20}$/;

    if (uname === '') {
        alert('Please enter your first name.');
        return;
    }

    if (email === '') {
        alert('Please enter your email address.');
        return;
    }

    if (!pattern.test(psswd)) {
        alert('Password must contain at least one digit, one lowercase and one uppercase letter, and be between 6 to 20 characters long.');
        return;
    }

    if (psswd === '') {
        alert('Please enter a password.');
        return;
    }

    if (repsswd === '') {
        alert('Please confirm your password.');
        return;
    }

    if (psswd !== repsswd) {
        alert('Passwords do not match.');
        return;
    }

    // If all validation passes, submit the form
    document.querySelector('form').submit();
});