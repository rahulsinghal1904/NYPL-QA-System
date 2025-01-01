css = '''
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    border: 1px solid #d4c19c; /* Subtle border to mimic aged parchment feel */
}
.chat-message.user {
    align: right;
    background-color: #d9cdbf; /* Warm beige for user messages */
}
.chat-message.bot {
    background-color: #a89276; /* Earthy brown tone for bot messages */
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid #d4c19c; /* Adds a subtle vintage-style border */
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #4b3e2a; /* Dark brown text for readability and classic feel */
    font-family: 'Georgia', serif; /* Old-school serif font for a vintage touch */
}
</style>
'''
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/6134/6134346.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message" style="text-align:right">{{MSG}}</div>
    <div class="avatar">
        <img src="https://img.freepik.com/premium-vector/cute-panda-is-thinking-vector-illustration-flat-cartoon-style_587427-106.jpg?w=1060">
    </div>    
    
</div>
'''