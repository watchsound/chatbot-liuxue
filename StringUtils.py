import re
import io


class StringUtils:
    @staticmethod
    def isEmpty(content):
        return content is None or len(content.strip()) == 0

    @staticmethod
    def checkNumberRow(content):
        pattern = r"^([0-9A-Za-z]{1,4})\.([\s]*).+"
        match = re.match(pattern, content)
        if match:
            n = match.group(1)
            try:
                return int(n)
            except ValueError:
                if len(n) == 1:
                    nc = n[0]
                    if nc.isalpha() and nc.isupper():
                        return ord(nc) - ord('A')
                    if nc.isalpha() and nc.islower():
                        return ord(nc) - ord('a')
        return -1

    @staticmethod
    def stripLeadingNumberOrder(content):
        pattern = r"^([0-9A-Za-z]{1,2})\.([\s]*)(.+)"
        match = re.match(pattern, content)
        if match:
            return match.group(3)
        return content

    @staticmethod
    def hasNumberOrder(lines, minCount):
        preNum = -1
        for line in lines:
            n = StringUtils.checkNumberRow(line)
            if n > preNum:
                minCount -= 1
                preNum = n
        return minCount <= 0

    @staticmethod
    def parseAsNumberedParagrah(lines, numCount):
        if not StringUtils.hasNumberOrder(lines, numCount):
            return None
        title = ""
        body = []
        sb = ""
        inHeader = True
        for line in lines:
            if len(line.strip()) == 0:
                continue
            n = StringUtils.checkNumberRow(line)
            if n >= 0:
                if inHeader:
                    title = sb
                else:
                    body.append(sb)
                sb = line
                inHeader = False
            else:
                if len(sb) > 0:
                    sb += "\n"
                sb += line
        return (title, body)

    @staticmethod
    def fileToString(file):
        try:
            return StringUtils.fileToString(io.open(file, "r", encoding="utf-8"))
        except Exception as ex:
            pass
        return ""

    @staticmethod
    def fileToString(in_stream):
        reader = io.StringIO(in_stream)
        writer = io.StringIO()
        buf = reader.read(1024)
        while buf:
            writer.write(buf)
            buf = reader.read(1024)
        return writer.getvalue()

    @staticmethod
    def stringToFile(text, file):
        with io.open(file, "w", encoding="utf-8") as writer:
            reader = io.StringIO(text)
            buf = reader.read(1000)
            while buf:
                writer.write(buf)
                buf = reader.read(1000)
            writer.flush()
