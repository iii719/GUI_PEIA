window.onload = (event) => {
    document.querySelectorAll('audio').forEach((audioElement, index) => {
        let playTime = 0;

        const updatePlayTime = () => {
            playTime = audioElement.currentTime;
        };

        const checkAndSetImplicitLike = (eventLabel) => {
            if (playTime >= 30) {
                document.getElementById(`implicit_like_${index + 1}`).value = 'true';
            }
        };

        audioElement.addEventListener('timeupdate', updatePlayTime);
        audioElement.addEventListener('ended', () => checkAndSetImplicitLike('Final playtime'));
        audioElement.addEventListener('pause', () => checkAndSetImplicitLike('Pause'));
    });
};
