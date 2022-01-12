import torch

def find_positions(ignore_specials, ignore_stopwords, the_tokens, stop_words):
    dot_positions = {}
    stopwords_positions = {}
    tmp = []

    if ignore_specials:
        word_counter = 0
        start_pointer = 0
        positions = {}

        num_of_tokens = len( the_tokens )
        num_of_tokens_range = range( num_of_tokens + 1 )

    else:
        word_counter = 1
        start_pointer = 1
        positions = {0: [0, 1]}

        num_of_tokens = len( the_tokens ) - 1
        num_of_tokens_range = range( 1, num_of_tokens + 1 )


    for i in num_of_tokens_range:

        if i == num_of_tokens:
            positions[word_counter] = [start_pointer, i]
            break

        if the_tokens[i][0] in ['Ġ', '.']:

            if ignore_stopwords:
                joined_tmp = "".join(tmp)
                current_word = joined_tmp[1:] if joined_tmp[0] == "Ġ" else joined_tmp
                if current_word in stop_words:
                    stopwords_positions[word_counter] = i-1

            if the_tokens[i] == ".":
                dot_positions[word_counter+1] = i

            positions[word_counter] = [start_pointer, i]
            word_counter += 1
            start_pointer = i
            tmp = []

        tmp.append(the_tokens[i])

    if not ignore_specials:
        positions[len( positions )] = [i, i+1]
    
    return positions, dot_positions, stopwords_positions

def make_the_words(inp, positions, ignore_specials):
    num_of_words = len( positions )

    if ignore_specials:
        the_words = inp.replace(".", " .").split(" ")[0:num_of_words]

    else:
        the_words = inp.replace(".", " .").split(" ")[0:(num_of_words-2)]
        the_words = ['[BOS]'] + the_words + ['[EOS]']
    
    return the_words

def scale(x, min_, max_):
    return (x - min_) / (max_ - min_)

def make_html(the_words, positions, final_score, num_words=15):
    the_html = ""

    for i, word in enumerate( the_words ):
        if i in positions:
            start = positions[i][0]
            end   = positions[i][1]

            if end - start > 1:
                score = torch.max( final_score[start:end] )
            else:
                score = final_score[start]

            the_html += """<span style="background-color:rgba(255, 0, 0, {});
                        padding:3px 6px 3px 6px; margin: 0px 2px 0px 2px" title="{}">{}</span>""" \
                        .format(score, score, word)

        if ((i+1) % num_words) == 0:
            the_html += "<br />"

    return the_html

def get_sample_article():
    return """sebastian vettel is determined to ensure the return of a long-standing ritual at ferrari is not a one-off this season. fresh from ferrari's first victory in 35 grands prix in malaysia 11 days ago, and ending his own 20-race drought, vettel returned to a hero's welcome at the team's factory at maranello last week. the win allowed ferrari to revive a tradition not seen at their base for almost two years since their previous triumph in may 2013 at the spanish grand prix courtesy of fernando alonso. sebastian vettel reflected on his stunning win for ferrari at the malaysian grand prix during the press conference before the weekend's chinese grand prix in shanghai the four-time world champion shares a friendly discussion with mclaren star jenson button four-times world champion vettel said: 'it was a great victory we had in malaysia, great for us as a team, and for myself a very emotional day - my first win with ferrari. 'when i returned to the factory on wednesday, to see all the people there was quite special. there are a lot of people working there and as you can imagine they were very, very happy. 'the team hadn't won for quite a while, so they enjoyed the fact they had something to celebrate. there were a couple of rituals involved, so it was nice for them to get that feeling again.' asked as to the specific nature of the rituals, vettel replied: 'i was supposed to be there for simulator work anyway, but it was quite nice to receive the welcome after the win. ferrari's vettel and britta roeske arrive at the shanghai circuit along with a ferrari mechanic, vettel caught up with members of his old team red bull on thursday 'all the factory got together for a quick lunch. it was quite nice to have all the people together in one room - it was a big room! - so we were able to celebrate altogether for a bit. 'i also learned when you win with ferrari, at the entry gate, they raise a ferrari flag - and obviously it's been a long time since they last did that. 'some 10 years ago there were a lot of flags, especially at the end of a season, so this flag will stay there for the rest of the year. 'we will, of course, try and put up another one sometime soon.' inside the ferrari garage, vettel shares a discussion with team staff as he looks to build on his sepang win ferrari team principal maurizio arrivabene shares a conversation with vettel at the team's hospitality suite the feeling is that will not happen after this weekend's race in china as the conditions at the shanghai international circuit are expected to suit rivals mercedes. not that vettel believes his success will be a one-off, adding: 'for here and the next races, we should be able to confirm we have a strong package and a strong car. 'we will want to make sure we stay ahead of the people we were ahead of in the first couple of races, but obviously knowing mercedes are in a very, very strong position. 'in general, for the start of a season things can be up and down, and we want to make sure there is quite a lot of ups, not so many downs. 'but it's normal in some races you are more competitive than others. 'we managed to do a very good job in malaysia, but for here and the next races we have to be realistic about we want to achieve.' ferrari mechanics show their joy after vettel won the malaysian grand prix, helping record the team's first formula one win since 2013 at the spanish grand prix"""