

from oac.app import ChatApp, ChatSession


def chat(sess: ChatSession):
    sess.system = """
        You are an exceptional programming assistant. Your
        task is to produce examples of poor coding in a variety
        of languages.
        Example references for analysis:
        * Buggy and syntactically incorrect code.
        * Code that is messy and does not comply with formatting standards and linting.
        * Code that is not clean as defined by Robert Martin, not SOLID.
        * Code that contains anti-patterns.

        Provide each code example followed by a description of the problem.
    """  # dedented internally

    for temperature in range(6, 18, 2):
        with sess as chat:
            chat.options.temperature = temperature / 10
            sess.user = "Create 5 code examples"



if __name__ == '__main__':
    ChatApp().start(chat)
