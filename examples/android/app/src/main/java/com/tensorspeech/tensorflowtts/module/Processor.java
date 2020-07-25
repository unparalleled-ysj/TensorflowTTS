package com.tensorspeech.tensorflowtts.module;


import android.util.Log;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * @author {@link "mailto:yusufsarigoz@gmail.com" "M. Yusuf Sarıgöz"}
 * Created 2020-07-25 17:25
 *
 * TODO Code for expanding numbers in the original Python code has not been implemented in Java yet.
 */
public class Processor {

    public static final String TAG = "processor";

    public static final String[] VALID_SYMBOLS = new String[] {
            "AA",
            "AA0",
            "AA1",
            "AA2",
            "AE",
            "AE0",
            "AE1",
            "AE2",
            "AH",
            "AH0",
            "AH1",
            "AH2",
            "AO",
            "AO0",
            "AO1",
            "AO2",
            "AW",
            "AW0",
            "AW1",
            "AW2",
            "AY",
            "AY0",
            "AY1",
            "AY2",
            "B",
            "CH",
            "D",
            "DH",
            "EH",
            "EH0",
            "EH1",
            "EH2",
            "ER",
            "ER0",
            "ER1",
            "ER2",
            "EY",
            "EY0",
            "EY1",
            "EY2",
            "F",
            "G",
            "HH",
            "IH",
            "IH0",
            "IH1",
            "IH2",
            "IY",
            "IY0",
            "IY1",
            "IY2",
            "JH",
            "K",
            "L",
            "M",
            "N",
            "NG",
            "OW",
            "OW0",
            "OW1",
            "OW2",
            "OY",
            "OY0",
            "OY1",
            "OY2",
            "P",
            "R",
            "S",
            "SH",
            "T",
            "TH",
            "UH",
            "UH0",
            "UH1",
            "UH2",
            "UW",
            "UW0",
            "UW1",
            "UW2",
            "V",
            "W",
            "Y",
            "Z",
            "ZH"
    };

    public static final String PAD = "_";
    public static final String EOS = "~";
    public static final String PUNCTUATION = "!'(),.:;? ";
    public static final String SPECIAL = "-";
    public static final String LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    public static final String[] ARPABET = new String[VALID_SYMBOLS.length];
    public static final List<String> SYMBOLS = new ArrayList(Arrays.asList(new String[] {PAD, SPECIAL}));
    public static final Map<String, Integer> SYMBOL_TO_ID = new HashMap<>();

    public static final Pattern CURLY_RE = Pattern.compile("(.*?)\\{(.+?)\\}(.*)");

    public static final Map<String, String> ABBREVIATIONS = new HashMap<>();

    static {
        Arrays.setAll(ARPABET, i -> "@" + VALID_SYMBOLS[i]);
        for (String p: PUNCTUATION.split("")) {
            if(!p.equals("")) {
                SYMBOLS.add(p);
            }
        }

        for (String l: LETTERS.split("")) {
            if(!l.equals("")) {
                SYMBOLS.add(l);
            }
        }

        SYMBOLS.addAll(Arrays.asList(ARPABET));
        SYMBOLS.add(EOS);

        for (int i = 0; i < SYMBOLS.size(); ++i) {
            SYMBOL_TO_ID.put(SYMBOLS.get(i), i);
        }

        ABBREVIATIONS.put("mrs", "misess");
        ABBREVIATIONS.put("mr", "mister");
        ABBREVIATIONS.put("dr", "doctor");
        ABBREVIATIONS.put("st", "saint");
        ABBREVIATIONS.put("co", "company");
        ABBREVIATIONS.put("jr", "junior");
        ABBREVIATIONS.put("maj", "major");
        ABBREVIATIONS.put("gen", "general");
        ABBREVIATIONS.put("drs", "doctors");
        ABBREVIATIONS.put("rev", "reverend");
        ABBREVIATIONS.put("lt", "lieutenant");
        ABBREVIATIONS.put("hon", "honorable");
        ABBREVIATIONS.put("sgt", "sergeant");
        ABBREVIATIONS.put("capt", "captain");
        ABBREVIATIONS.put("esq", "esquire");
        ABBREVIATIONS.put("ltd", "limited");
        ABBREVIATIONS.put("col", "colonel");
        ABBREVIATIONS.put("ft", "fort");
    }


    public static List<Integer> symbolsToSequence(String symbols) {
        List<Integer> sequence = new ArrayList<>();

        for (int i = 0; i < symbols.length(); ++i) {
            sequence.add(SYMBOL_TO_ID.get(String.valueOf(symbols.charAt(i))));
        }

            return sequence;
    }

    public static List<Integer> arpabetToSequence(String symbols) {
        List<Integer> sequence = new ArrayList<>();
        String[] as = symbols.split(" ");
        for (String s : as) {
            sequence.add(SYMBOL_TO_ID.get("@" + s));
        }
        return sequence;
    }

    public static String convertToAscii(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.US_ASCII);
        return new String(bytes);
    }

    public static String collapseWhitespace(String text) {
        return text.replaceAll("\\s+", " ");
    }

    public static String expandAbbreviations(String text) {
        for (Map.Entry<String, String> entry : ABBREVIATIONS.entrySet()) {
            text = text.replaceAll("\\b" + entry.getKey() + "\\.", entry.getValue());
        }
        return text;
    }

    public static String cleanTextForEnglish(String text) {
        text = convertToAscii(text);
        text = text.toLowerCase();
        text = expandAbbreviations(text);
        text = collapseWhitespace(text);
        return text;
    }

    public static int[] textToIds(String text) {
        List<Integer> sequence = new ArrayList<>();
        while (text.length() > 0) {
            Matcher m = CURLY_RE.matcher(text);
            if(!m.find()) {
                sequence.addAll(symbolsToSequence(cleanTextForEnglish(text)));
                break;
            }
            sequence.addAll(symbolsToSequence(cleanTextForEnglish(m.group(1))));
            sequence.addAll(arpabetToSequence(m.group(2)));
            text = m.group(3);
        }

        int size = sequence.size();
        Integer[] tmp = new Integer[size];
        tmp = sequence.toArray(tmp);
        int[] ids = new int[size];
        for(int i = 0; i < size; ++i) {
            ids[i] = Integer.parseInt(tmp[i].toString());
        }
        return ids;
    }
}
